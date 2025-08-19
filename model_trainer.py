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

class SaveEveryNStepsCallback(TrainerCallback):
    """
    Un callback personalizzato per salvare il modello e il tokenizer
    ogni N passi di addestramento.
    """
    def __init__(self, output_dir, save_steps=500):
        super().__init__()
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_saved_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > self.last_saved_step:
            try:
                # Controlla se il modello è un PeftModel e salva la versione PEFT
                if isinstance(kwargs['model'], PeftModel):
                    output_checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
                    kwargs['model'].save_pretrained(output_checkpoint_dir)
                    kwargs['tokenizer'].save_pretrained(output_checkpoint_dir)
                    progress_container(None, f"Checkpoint salvato al passo {state.global_step}", "info")
                    self.last_saved_step = state.global_step
            except Exception as e:
                progress_container(None, f"Errore nel salvataggio del checkpoint al passo {state.global_step}: {e}", "error")

class LossLoggingCallback(TrainerCallback):
    """
    Un callback personalizzato per stampare la loss ogni N passi
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        # I log di loss sono disponibili nell'evento on_log
        if logs is not None and "loss" in logs:
            progress_container(None, f"Passo {state.global_step}: Loss = {logs['loss']:.4f}", "info")
            
# Funzione ausiliaria per la creazione di messaggi di progresso
def progress_container(status_placeholder, message, type="info"):
    """
    Funzione mock per simulare l'aggiornamento di un placeholder Streamlit.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)

# ==============================================================================
# SEZIONE 3: FUNZIONE PRINCIPALE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, progress_container):
    """
    Esegue il fine-tuning del modello di linguaggio.
    
    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente il corpus di addestramento.
        progress_container (function): Funzione per visualizzare i messaggi di progresso.
    
    Returns:
        tuple: Una tupla contenente il modello e il tokenizer fine-tuned.
    """
    try:
        progress_container("Inizializzazione del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        progress_container("Preparazione dei dati...", "info")
        # Inizializza i set di dati
        data = Dataset.from_pandas(corpus_df)
        dataset_dict = DatasetDict({
            'train': data.select(range(int(len(data) * 0.9))),
            'eval': data.select(range(int(len(data) * 0.1)))
        })
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['eval']
        
        progress_container("Caricamento del modello pre-addestrato...", "info")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")
        
        progress_container("Configurazione di PEFT (Parameter-Efficient Fine-Tuning)...", "info")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        peft_model = get_peft_model(model, peft_config)
        
        progress_container("Configurazione degli argomenti di addestramento...", "info")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{OUTPUT_DIR}/logs',
            logging_steps=100,
            report_to="none",
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
        )
        
        # Cerca l'ultimo checkpoint salvato per riprendere l'addestramento
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
            for d in sorted(dirs, reverse=True):
                if d.startswith("checkpoint-"):
                    last_checkpoint = os.path.join(OUTPUT_DIR, d)
        
        progress_container("Avvio del processo di addestramento...", "info")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[SaveEveryNStepsCallback(output_dir=OUTPUT_DIR, save_steps=500), LossLoggingCallback()]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Step 5: Salva il modello finale
        progress_container("Addestramento completato. Salvataggio del modello finale...", "info")
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        
        # Salva il modello e il tokenizer
        peft_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        # Salva lo stato del trainer
        trainer.save_state()
        
        progress_container(f"Modello e tokenizer finali salvati in: {final_model_path}", "success")
        return peft_model, tokenizer
    
    except Exception as e:
        progress_container(f"Errore critico in model_trainer.py: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        raise
