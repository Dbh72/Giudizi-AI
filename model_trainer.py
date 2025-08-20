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
            print(f"Step: {state.global_step}, Loss: {state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else 'N/A'}")
            
# ==============================================================================
# SEZIONE 3: FUNZIONI PER LA PREPARAZIONE DEI DATI E L'ADDESTRAMENTO
# ==============================================================================

def preprocess_function(examples, tokenizer):
    """
    Funzione per tokenizzare gli input e i target per il fine-tuning.
    """
    # Combina la descrizione e il giudizio per il fine-tuning.
    # Aggiungiamo un prefisso per indicare al modello che deve generare il giudizio.
    inputs = [f"Descrizione: {desc}" for desc in examples["Descrizione"]]
    targets = [f"{giudizio}" for giudizio in examples["Giudizio"]]
    
    # Tokenizzazione degli input
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Tokenizzazione dei target (con gestione speciale)
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    # Sostituiamo -100 con l'ID del pad token per calcolare correttamente la loss.
    labels_with_ignore_index = []
    for label in labels["input_ids"]:
        labels_with_ignore_index.append([t if t != tokenizer.pad_token_id else -100 for t in label])

    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

def train_model(corpus_df, progress_container):
    """
    Avvia il processo di fine-tuning del modello.
    """
    progress_container("Preparazione dei dati per l'addestramento...", "info")
    
    try:
        # Seleziona solo le colonne necessarie
        corpus_df = corpus_df[['Descrizione', 'Giudizio']]
        
        # Converte il DataFrame in un oggetto Dataset
        dataset = Dataset.from_pandas(corpus_df)
        
        # Divide il dataset in training e validation set (80/20)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Carica il tokenizer e il modello pre-addestrato
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        # Configura l'adattatore LoRA (Low-Rank Adaptation)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # Tokenizza il dataset
        tokenized_dataset = split_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        
        # Configura gli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        # Data Collator per il padding dinamico
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
        
        # Inizializza il Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            callbacks=[LossLoggingCallback()]
        )
        
        progress_container("Inizio del fine-tuning del modello. Questo potrebbe richiedere tempo...", "info")
        
        # Avvia l'addestramento
        trainer.train()
        
        progress_container("Fine-tuning completato. Salvataggio del modello...", "info")
        
        # Salva il modello fine-tuned
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        os.makedirs(final_model_dir, exist_ok=True)
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        progress_container(f"Modello fine-tuned salvato in: {final_model_dir}", "success")
        
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
