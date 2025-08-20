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
            if "loss" in state.log_history[-1]:
                print(f"Passo {state.global_step}: Loss = {state.log_history[-1]['loss']:.4f}")

# ==============================================================================
# SEZIONE 3: FUNZIONI PER L'ADDESTRAMENTO
# ==============================================================================

def train_model(corpus_df, progress_container):
    """
    Addestra il modello di linguaggio utilizzando il corpus di addestramento.
    """
    progress_container("Avvio dell'addestramento del modello...", "info")
    
    # Crea la directory di output se non esiste
    output_dir = os.path.join(OUTPUT_DIR, "checkpoint")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        progress_container("Preparazione dei dati per l'addestramento...", "info")
        
        # Converti il DataFrame in un oggetto Dataset di Hugging Face
        dataset = Dataset.from_pandas(corpus_df)
        
        # Suddividi il dataset in training e evaluation (al 90% per l'addestramento)
        # Nota: in questo caso, non essendoci un set di valutazione separato,
        # la logica di 'evaluation_strategy' non è strettamente necessaria.
        # dataset_dict = dataset.train_test_split(test_size=0.1)
        dataset_dict = DatasetDict({
            'train': dataset
        })
        
        progress_container("Caricamento del tokenizer e del modello base...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        # Configura l'addestramento LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        model = get_peft_model(base_model, peft_config)
        
        progress_container("Tokenizzazione del dataset...", "info")
        # Funzione di tokenizzazione
        def tokenize_function(examples):
            # Controlla la lunghezza massima supportata dal modello
            max_input_length = tokenizer.model_max_length
            
            # Tokenizza l'input e il target
            model_inputs = tokenizer(examples['input_text'], max_length=max_input_length, truncation=True)
            labels = tokenizer(examples['target_text'], max_length=max_input_length, truncation=True)
            
            # Assegna gli ID tokenizzati del target come labels
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
            
        # Mappa la funzione di tokenizzazione al dataset
        tokenized_datasets = dataset_dict.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        # Data collator per il padding dinamico
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

        # Configurazione degli argomenti di addestramento
        # NOTA: Abbiamo rimosso 'evaluation_strategy' e 'load_best_model_at_end'
        # per risolvere il TypeError e semplificare il processo dato che
        # non è stato fornito un set di valutazione.
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_strategy='steps',
            logging_steps=100,
            save_strategy='epoch',
            save_steps=100,
            report_to="none"
        )
        
        # Inizializza il Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[LossLoggingCallback()]
        )
        
        # Avvia l'addestramento
        trainer.train()
        
        # Salva il modello fine-tuned
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        progress_container("Addestramento completato e modello salvato con successo!", "success")
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
        if not os.path.exists(model_path):
            progress_container(f"Errore: La directory del modello fine-tuned non esiste. Addestra prima un modello.", "error")
            return None, None
            
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
    model_path = os.path.join(OUTPUT_DIR, "final_model")
    if os.path.exists(model_path):
        try:
            shutil.rmtree(model_path)
            progress_container("Modello fine-tuned eliminato con successo.", "success")
        except Exception as e:
            progress_container(f"Errore nell'eliminazione del modello: {e}", "error")
    else:
        progress_container("Nessun modello da eliminare.", "warning")

