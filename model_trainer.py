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
    def __init__(self, progress_container):
        self.progress_container = progress_container
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            loss = state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else "N/A"
            self.progress_container(f"Passo {state.global_step}: Loss = {loss:.4f}", "info")

# ==============================================================================
# SEZIONE 3: FUNZIONI PER L'ADDESTRAMENTO
# ==============================================================================

def fine_tune_model(training_df, progress_container):
    """
    Esegue il fine-tuning del modello.
    """
    try:
        progress_container("Inizio processo di fine-tuning...", "info")
        
        # 1. Prepara il dataset
        progress_container("Preparazione del dataset...", "info")
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(training_df)
        })
        
        # 2. Carica il modello e il tokenizer
        progress_container("Caricamento del modello e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        # 3. Prepara il modello per il fine-tuning con PEFT (LoRA)
        progress_container("Configurazione dell'adattatore PEFT (LoRA)...", "info")
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
        
        # 4. Tokenizza il dataset
        progress_container("Tokenizzazione del dataset...", "info")
        def preprocess_function(examples):
            inputs = [f"generare un giudizio per il seguente testo: {text}" for text in examples["input_text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)

            # Configura il tokenizer per i target (labels)
            labels = tokenizer(text_target=examples["target_text"], max_length=512, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset_dict.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])

        # 5. Configura gli argomenti di addestramento
        progress_container("Configurazione degli argomenti di addestramento...", "info")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            learning_rate=3e-4,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_strategy="epoch",
            save_steps=100,
            seed=42,
            gradient_accumulation_steps=4,
        )

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            max_length=512,
        )

        # 6. Avvia l'addestramento
        progress_container("Addestramento del modello in corso...", "info")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            data_collator=data_collator,
            callbacks=[LossLoggingCallback(progress_container)]
        )
        trainer.train()

        # 7. Salva il modello fine-tuned
        progress_container("Salviamo il modello fine-tuned...", "info")
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        progress_container("Modello fine-tuned salvato con successo!", "success")
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

