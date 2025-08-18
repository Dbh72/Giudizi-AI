# ==============================================================================
# File: model_trainer.py
# Modulo per l'addestramento (Fine-Tuning) del modello di linguaggio.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo tutte le librerie essenziali per l'addestramento.
import torch
import warnings
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
import shutil
import traceback
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
import json

# Ignoriamo i FutureWarning per una console più pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE
# ==============================================================================

# Directory dove il modello addestrato verrà salvato.
OUTPUT_DIR = "modello_finetunato"
# Il nome del modello pre-addestrato di base da utilizzare.
MODEL_NAME = "google/flan-t5-base"

# ==============================================================================
# SEZIONE 3: CLASSI E CALLBACK PERSONALIZZATI
# ==============================================================================

class SaveEveryNStepsCallback(TrainerCallback):
    """
    Un callback personalizzato per salvare il modello e lo stato di addestramento
    ogni N passi.
    """
    def __init__(self, output_dir, save_steps=500):
        self.output_dir = output_dir
        self.save_steps = save_steps
        os.makedirs(self.output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            print(f"Salvataggio del checkpoint al passo {state.global_step}...")
            # Salva il modello PEFT
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            # Salva il tokenizer
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            # Salva lo stato del trainer
            state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))
            print(f"Checkpoint salvato in: {checkpoint_dir}")
            
# ==============================================================================
# SEZIONE 4: FUNZIONE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, fine_tuning_state):
    """
    Esegue il fine-tuning di un modello di linguaggio utilizzando i dati del corpus.

    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente il corpus di addestramento.
        fine_tuning_state (dict): Un dizionario di stato per aggiornare i progressi.

    Returns:
        str: Il percorso della directory del modello fine-tuned.
    """
    try:
        # Checkpoint 1: Preparazione e tokenizzazione del dataset.
        print("Avvio del fine-tuning del modello...")
        print("Checkpoint 1: Preparazione e tokenizzazione del dataset.")
        
        # Inizializza tokenizer e modello
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Prepara il dataset Hugging Face
        dataset = Dataset.from_pandas(corpus_df)
        
        # Tokenizza i dati
        def tokenize_function(examples):
            # Aggiungi il prefisso "train: " per il fine-tuning di un modello T5
            inputs = [f"giudizio: {text}" for text in examples["input_text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)
            labels = tokenizer(examples["target_text"], max_length=512, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Checkpoint 2: Caricamento del modello base e configurazione PEFT.
        print("Checkpoint 2: Caricamento del modello base e configurazione PEFT.")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()

        # Checkpoint 3: Configurazione degli argomenti di addestramento.
        print("Checkpoint 3: Configurazione degli argomenti di addestramento.")
        
        # Verifica se esiste un checkpoint precedente
        last_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            checkpoints = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith("checkpoint-")]
            if checkpoints:
                # Trova il checkpoint con il numero di step più alto
                last_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoints, key=lambda x: int(x.split('-')[1]))[-1])
                print(f"Trovato checkpoint precedente: {last_checkpoint}. Riprendo l'addestramento.")
        
        # Rimuoviamo gli argomenti che causano l'errore
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=500,
            save_steps=500,
            logging_steps=500, # Continua a loggare i progressi
            report_to="none", # Disabilita i log a servizi esterni
            fp16=torch.cuda.is_available(), # Usa fp16 se la GPU è disponibile
            push_to_hub=False # Non caricare il modello su Hugging Face Hub
        )

        # Step 4: Avvia il trainer
        print("Avvio del processo di addestramento...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[SaveEveryNStepsCallback(output_dir=OUTPUT_DIR)]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Step 5: Salva il modello finale
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        peft_model.save_pretrained(final_model_path)
        print(f"Addestramento completato. Modello salvato in: {final_model_path}")

        # Salva il tokenizer
        tokenizer.save_pretrained(final_model_path)
        print(f"Tokenizer salvato in: {final_model_path}")

        # Salva lo stato di addestramento
        trainer.save_state()
        print(f"Stato di addestramento salvato.")

        return final_model_path

    except Exception as e:
        print(f"Errore durante il fine-tuning: {e}")
        print("Traceback:", traceback.format_exc())
        fine_tuning_state.update({"status": f"Errore: {e}", "progress": 0.0, "current_step": "Errore durante l'addestramento."})
        raise e
