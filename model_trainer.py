# ==============================================================================
# File: model_trainer.py
# Modulo per l'addestramento (Fine-Tuning) del modello di linguaggio.
# Questo file contiene la logica per preparare i dati, configurare l'ambiente
# di addestramento e avviare il processo di fine-tuning.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo tutte le librerie essenziali per l'addestramento.
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

class LossLoggingCallback(TrainerCallback):
    """
    Callback per stampare la loss di addestramento ogni 100 passi.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            loss = kwargs.get('loss')
            if loss is not None:
                print(f"Step {state.global_step}: Loss = {loss:.4f}")

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def train_model(corpus_df, progress_container):
    """
    Addestra il modello di linguaggio utilizzando il corpus fornito.

    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente i dati di addestramento.
        progress_container (callable): Funzione per inviare messaggi di stato.
    
    Returns:
        tuple: Il modello e il tokenizer addestrati, o None, None in caso di errore.
    """
    try:
        if corpus_df.empty:
            progress_container("Corpus di addestramento vuoto. Addestramento annullato.", "warning")
            return None, None
            
        progress_container(f"Avvio del fine-tuning del modello '{MODEL_NAME}'...", "info")

        # 1. Preparazione del Dataset
        progress_container("Preparazione del dataset...", "info")
        dataset = Dataset.from_pandas(corpus_df)
        dataset = dataset.train_test_split(test_size=0.1) # Split del 10% per test
        
        progress_container("Dataset creato e diviso in set di addestramento e test.", "success")
        
        # 2. Caricamento del Modello e del Tokenizer
        progress_container("Caricamento del modello base e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        progress_container("Modello e tokenizer caricati. Configurazione per PEFT...", "success")

        # 3. Configurazione PEFT (Parameter-Efficient Fine-Tuning)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        
        # 4. Ottenere il modello PEFT
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        progress_container("Modello configurato per il fine-tuning efficiente (PEFT).", "success")

        # 5. Tokenizzazione dei dati
        def preprocess_function(examples):
            inputs = [ex for ex in examples["input_text"]]
            targets = [ex for ex in examples["target_text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

        progress_container("Dati tokenizzati con successo.", "success")
        
        # 6. Configurazione degli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps", # Strategia di valutazione basata sui passi
            eval_steps=500, # Esegue la valutazione ogni 500 passi
            save_strategy="steps", # Strategia di salvataggio basata sui passi
            save_steps=500, # Salva un checkpoint ogni 500 passi
            save_total_limit=3, # Limita il numero di checkpoint salvati
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            num_train_epochs=3,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=100,
            report_to="none" # Disabilita i report a servizi esterni
        )

        # 7. Inizializzazione del Data Collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

        # 8. Creazione del Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            callbacks=[LossLoggingCallback()]
        )

        progress_container("Trainer creato. Avvio dell'addestramento...", "info")
        
        # 9. Avvio dell'addestramento
        trainer.train()

        progress_container("Addestramento completato con successo. Salvataggio del modello finale...", "info")

        # 10. Salvataggio del modello fine-tuned
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.save_model(final_model_dir)

        progress_container("Modello finale e tokenizer salvati.", "success")

        return model, tokenizer

    except Exception as e:
        progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

def load_fine_tuned_model(progress_container):
    """
    Carica un modello fine-tuned e il suo tokenizer da una directory.
    """
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        progress_container(f"Caricamento del modello fine-tuned da: {model_path}...", "info")
        
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carica il modello base
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        
        # Carica l'adattatore PEFT
        model = PeftModel.from_pretrained(base_model, model_path)
        
        progress_container("Modello e tokenizer caricati con successo.", "success")
        return model, tokenizer
        
    except Exception as e:
        progress_container(f"Errore nel caricamento del modello: {e}. Il modello potrebbe non essere stato ancora addestrato.", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None
        
def delete_model(progress_container):
    """
    Elimina la directory del modello fine-tuned.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        progress_container("Modello fine-tuned eliminato.", "success")
    else:
        progress_container("Nessun modello da eliminare.", "warning")
