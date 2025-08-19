# ==============================================================================
# File: model_trainer.py
# Modulo per l'addestramento (Fine-Tuning) del modello di linguaggio.
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

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            peft_model = kwargs['model']
            tokenizer = kwargs['tokenizer']
            
            # Crea un percorso di salvataggio basato sul passo corrente
            save_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            peft_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Modello e tokenizer salvati al passo {state.global_step} in: {save_path}")

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def find_last_checkpoint(output_dir):
    """
    Trova l'ultimo checkpoint salvato nella directory di output.
    """
    if not os.path.exists(output_dir):
        return None
    
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    # Ordina i checkpoint in base al numero di passo
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    
    return os.path.join(output_dir, checkpoints[-1])

def fine_tune_model(train_df, eval_df, progress_container):
    """
    Funzione principale per il fine-tuning del modello.
    """
    try:
        # Step 1: Prepara i dati in formato Dataset
        progress_container("Preparazione dei dati per il fine-tuning...", "info")
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)

        # Step 2: Carica il tokenizer e il modello base
        progress_container(f"Caricamento del modello base '{MODEL_NAME}' e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

        # Step 3: Configura e applica LoRA
        progress_container("Configurazione di LoRA per il fine-tuning...", "info")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        
        last_checkpoint = find_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            progress_container(f"Trovato checkpoint precedente: {last_checkpoint}. Riprendendo l'addestramento...", "info")
            peft_model = PeftModel.from_pretrained(peft_model, last_checkpoint)

        # Step 4: Avvia l'addestramento
        progress_container("Configurazione degli argomenti di addestramento...", "info")
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            # L'argomento 'evaluation_strategy' è stato rinominato in 'eval_strategy'
            # per essere compatibile con le versioni più recenti di Transformers.
            # Questo risolve l'errore che hai riscontrato.
            eval_strategy="steps", 
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none" # Disabilita i report per evitare dipendenze aggiuntive
        )
   
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
        return peft_model, tokenizer
    
    except Exception as e:
        progress_container(f"Errore critico in model_trainer.py: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        # Rimuove la directory del modello se l'addestramento fallisce
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        return None, None
