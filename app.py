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
import streamlit as st
import glob

# Ignoriamo i FutureWarning per una console più pulita.
warnings.filterwarnings("ignore")

# Definiamo le costanti per il progetto
OUTPUT_DIR = "./modello_finetunato"
MODEL_NAME = "t5-small"

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
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            output_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(output_path)
            kwargs['tokenizer'].save_pretrained(output_path)
            control.should_save = True

class LossLoggingCallback(TrainerCallback):
    """
    Callback per loggare le metriche di addestramento e valutazione.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            loss = logs["loss"]
            if "eval_loss" in logs:
                eval_loss = logs["eval_loss"]
                message = f"Step: {step}, Loss: {loss:.4f}, Eval Loss: {eval_loss:.4f}"
            else:
                message = f"Step: {step}, Loss: {loss:.4f}"
            
            # Utilizziamo la funzione di logging definita nell'app.py
            progress_container = st.session_state.get('progress_container')
            if progress_container:
                progress_container(st.empty(), message, "info")

# ==============================================================================
# SEZIONE 3: FUNZIONI DI PRE-ELABORAZIONE DEI DATI
# ==============================================================================

def progress_container_stub(*args, **kwargs):
    """Funzione placeholder per evitare errori se non in ambiente Streamlit."""
    pass

def preprocess_function(examples, tokenizer, max_length=512):
    """
    Pre-elabora gli esempi per il fine-tuning.
    Implementa la logica di chunking e overflow per gestire testi lunghi.
    
    Args:
        examples: Il batch di esempi dal dataset.
        tokenizer: Il tokenizer del modello.
        max_length (int): La lunghezza massima della sequenza.
    
    Returns:
        Un dizionario con le sequenze tokenizzate e la logica di overflow.
    """
    # Tokenizza le sorgenti e i target
    model_inputs = tokenizer(
        examples["source"], 
        max_length=max_length, 
        truncation=False, # Imposta su False per gestire il truncation manualmente
        return_overflowing_tokens=True,
        padding="max_length"
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"], 
            max_length=max_length, 
            truncation=True,
            padding="max_length"
        )
    
    # La logica di "chunking" con "overflow" viene gestita qui.
    # Il tokenizer stesso, quando return_overflowing_tokens=True, produce
    # più sequenze per un singolo esempio di input troppo lungo.
    # Assicuriamo che ogni input abbia una label corrispondente.
    sample_map = model_inputs.pop("overflow_to_sample_mapping")
    model_inputs["labels"] = []
    
    for i in range(len(model_inputs["input_ids"])):
        sample_index = sample_map[i]
        model_inputs["labels"].append(labels["input_ids"][sample_index])

    return model_inputs

# ==============================================================================
# SEZIONE 4: FUNZIONE PRINCIPALE PER L'ADDESTRAMENTO
# ==============================================================================

def fine_tune_model(corpus_file_path, status_placeholder):
    """
    Carica un modello pre-addestrato, lo fine-tuna su un corpus dato e lo salva.
    
    Args:
        corpus_file_path (str): Il percorso al file del corpus (CSV).
        status_placeholder: Il placeholder di Streamlit per mostrare i progressi.
        
    Returns:
        tuple: Il modello fine-tunato e il suo tokenizer.
    """
    try:
        # Step 1: Caricamento e preparazione dei dati
        progress_container = st.session_state.get('progress_container', progress_container_stub)
        progress_container(status_placeholder, "Caricamento e preparazione del dataset...", "info")
        
        df = pd.read_csv(corpus_file_path)
        dataset = Dataset.from_pandas(df)
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']

        # Step 2: Inizializzazione di tokenizer e modello
        progress_container(status_placeholder, "Inizializzazione di tokenizer e modello...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Step 3: Configurazione PEFT e LoRA
        progress_container(status_placeholder, "Configurazione di PEFT e LoRA...", "info")
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
        
        # Carica un checkpoint se esiste
        last_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            list_of_checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
            if list_of_checkpoints:
                latest_checkpoint_path = max(list_of_checkpoints, key=os.path.getctime)
                last_checkpoint = latest_checkpoint_path
                peft_model = PeftModel.from_pretrained(peft_model, last_checkpoint)
                progress_container(status_placeholder, f"Ripresa da checkpoint: {last_checkpoint}", "info")
        
        # Step 4: Tokenizzazione del dataset
        progress_container(status_placeholder, "Tokenizzazione del dataset...", "info")
        tokenized_train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_eval_dataset = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

        # Step 5: Configurazione dei parametri di addestramento
        progress_container(status_placeholder, "Configurazione dei parametri di addestramento...", "info")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            report_to="none" # Disabilita i report esterni
        )

        # Step 6: Avvio del processo di addestramento
        progress_container(status_placeholder, "Avvio del processo di addestramento...", "info")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[LossLoggingCallback()]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Step 7: Salva il modello finale
        progress_container(status_placeholder, "Addestramento completato. Salvataggio del modello finale...", "info")
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        peft_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        trainer.save_state()
        
        progress_container(status_placeholder, f"Modello e tokenizer finali salvati in: {final_model_path}", "success")
        return peft_model, tokenizer
    
    except Exception as e:
        progress_container(status_placeholder, f"Errore critico in model_trainer.py: {e}", "error")
        progress_container(status_placeholder, f"Traceback: {traceback.format_exc()}", "error")
        return None, None
