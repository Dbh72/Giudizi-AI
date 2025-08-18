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
    Callback personalizzato per salvare il modello ogni N passi,
    sovrascrivendo la cartella del modello PEFT.
    """
    def __init__(self, output_dir, save_steps=500):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_saved_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0 and state.global_step > self.last_saved_step:
            print(f"Salvataggio del modello al passo {state.global_step}...")
            # Salvataggio del modello PEFT e del tokenizer
            peft_model = kwargs.get('model')
            tokenizer = kwargs.get('tokenizer')
            
            output_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            peft_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            print(f"Modello salvato in: {output_path}")
            self.last_saved_step = state.global_step

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Funzione per la tokenizzazione dei dati di input e target.
    Aggiunge il token di fine sequenza al target.
    """
    model_inputs = tokenizer(examples['input_text'], max_length=max_length, truncation=True)
    labels = tokenizer(text_target=examples['target_text'], max_length=max_length, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# ==============================================================================
# SEZIONE 4: FUNZIONE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, fine_tuning_state):
    """
    Esegue il fine-tuning del modello di linguaggio.

    Args:
        corpus_df (pd.DataFrame): DataFrame contenente i dati di addestramento.
        fine_tuning_state (dict): Dizionario di stato per la gestione dei checkpoint.

    Returns:
        str: Il percorso della directory del modello salvato.
    """
    print("Avvio del fine-tuning del modello...")
    try:
        # Step 1: Prepara il dataset
        # Aggiungiamo il checkpointing logico qui
        print("Checkpoint 1: Preparazione e tokenizzazione del dataset.")
        
        # Converti il DataFrame in un Hugging Face Dataset
        dataset = Dataset.from_pandas(corpus_df)

        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        # Tokenizza il dataset
        tokenized_dataset = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

        # Step 2: Carica il modello base e configura PEFT
        # Aggiungiamo il checkpointing logico qui
        print("Checkpoint 2: Caricamento del modello base e configurazione PEFT.")
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()

        # Data Collator per il padding dinamico
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)
        
        # Step 3: Configura gli argomenti di addestramento
        # Aggiungiamo il checkpointing logico qui
        print("Checkpoint 3: Configurazione degli argomenti di addestramento.")
        
        # Impostiamo il percorso di output finale, rimuovendo la cartella se esiste
        final_output_dir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        if os.path.exists(final_output_dir):
            shutil.rmtree(final_output_dir)
        os.makedirs(final_output_dir, exist_ok=True)
        
        # Controlliamo se esiste uno stato di addestramento precedente
        resume_from_checkpoint = False
        if fine_tuning_state.get("last_checkpoint"):
            resume_from_checkpoint = fine_tuning_state["last_checkpoint"]
            print(f"Riprendo l'addestramento dal checkpoint: {resume_from_checkpoint}")
        
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            learning_rate=2e-4,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(), # Abilita l'accelerazione GPU solo se disponibile
            logging_dir='./logs',
            logging_steps=100,
            save_strategy="steps", # Strategia di salvataggio
            save_steps=500, # Salva ogni 500 step
            evaluation_strategy="steps", # Strategia di valutazione
            eval_steps=500, # Valuta ogni 500 step
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            gradient_checkpointing=True,
            report_to="none", # Non riportare a W&B o altri
            gradient_accumulation_steps=1,
            overwrite_output_dir=True,
            save_total_limit=3, # Limita il numero di checkpoint salvati
            resume_from_checkpoint=resume_from_checkpoint,
            disable_tqdm=False,
            do_eval=True # Aggiunto per garantire il corretto funzionamento di evaluation_strategy
        )

        # Step 4: Avvia il trainer
        # Aggiungiamo il checkpointing logico qui
        print("Checkpoint 4: Avvio del processo di addestramento.")
        
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[SaveEveryNStepsCallback(output_dir=final_output_dir)]
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Step 5: Salva il modello finale
        # Aggiungiamo il checkpointing logico qui
        print("Checkpoint 5: Salvataggio del modello finale.")
        
        final_model_path = os.path.join(final_output_dir, "final_model")
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
        print(f"Errore nel fine-tuning: {e}")
        print(traceback.format_exc())
        raise e
