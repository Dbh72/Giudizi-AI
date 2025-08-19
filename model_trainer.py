# ==============================================================================
# File: model_trainer.py
# Modulo per l'addestramento (Fine-Tuning) del modello di linguaggio.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import torch
import warnings
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
import shutil
import traceback
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
import json
import pandas as pd
from config import OUTPUT_DIR, MODEL_NAME

# Ignoriamo i FutureWarning per una console più pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CLASSI E CALLBACK PERSONALIZZATI
# ==============================================================================

class SaveEveryNStepsCallback(TrainerCallback):
    """
    Un callback personalizzato per salvare il modello e il tokenizer
    ogni N passi di addestramento.
    """
    def __init__(self, output_dir, save_steps=500):
        super().__init__()
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_checkpoint_path = None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            self.last_checkpoint_path = checkpoint_dir
            print(f"Salvato checkpoint intermedio a {checkpoint_dir}")

def find_last_checkpoint(output_dir):
    """
    Trova l'ultimo checkpoint salvato nella directory di output.
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    # Ordina i checkpoint per numero di passo e restituisce l'ultimo
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    return os.path.join(output_dir, checkpoints[-1])

def fine_tune(corpus_df, progress_container):
    """
    Esegue il fine-tuning di un modello Flan-T5 utilizzando il corpus fornito.
    """
    try:
        # Step 1: Prepara il dataset
        if 'input_text' not in corpus_df.columns or 'target_text' not in corpus_df.columns:
            progress_container("Errore: Il DataFrame del corpus deve contenere le colonne 'input_text' e 'target_text'.", "error")
            return False

        # Converte il DataFrame in un Dataset di Hugging Face
        dataset = Dataset.from_pandas(corpus_df)

        # Crea uno split di addestramento e validazione (80/20)
        split_dataset = dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
        
        progress_container("Dataset di addestramento preparato con successo.", "success")

        # Step 2: Carica il tokenizer e il modello di base
        progress_container(f"Caricamento del tokenizer e del modello di base '{MODEL_NAME}'...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Utilizziamo device_map="auto" per distribuire il modello automaticamente se è presente una GPU
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        
        # Aggiungi un token di padding se non esiste
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            base_model.resize_token_embeddings(len(tokenizer))
            progress_container("Token di padding aggiunto e embedding ridimensionati.", "info")

        # Step 3: Configura e applica PEFT (LoRa)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        peft_model = get_peft_model(base_model, lora_config)
        progress_container("Modello PEFT (LoRa) configurato e applicato.", "success")
        peft_model.print_trainable_parameters()
        
        # Cerca l'ultimo checkpoint per la ripresa
        last_checkpoint = find_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            progress_container(f"Trovato l'ultimo checkpoint: {last_checkpoint}. L'addestramento riprenderà da qui.", "info")
            
            # Carica lo stato del trainer
            trainer_state_file = os.path.join(last_checkpoint, "trainer_state.json")
            if os.path.exists(trainer_state_file):
                with open(trainer_state_file, 'r') as f:
                    trainer_state = json.load(f)
                    progress_container(f"Stato di addestramento caricato. Passo corrente: {trainer_state.get('global_step', 0)}", "info")

        # Definisce gli argomenti per l'addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Step 4: Avvia il trainer
        progress_container("Avvio del processo di addestramento...", "info")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[SaveEveryNStepsCallback(output_dir=OUTPUT_DIR, save_steps=500)]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Step 5: Salva il modello finale
        progress_container("Addestramento completato. Salvataggio del modello finale...", "info")
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        peft_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        trainer.save_state()
        
        progress_container(f"Modello e tokenizer finali salvati in: {final_model_path}", "success")
        return True

    except Exception as e:
        progress_container(f"Errore critico in `model_trainer.py`: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return False
