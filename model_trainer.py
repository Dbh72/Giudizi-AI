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
            output_checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(output_checkpoint_dir)
            if kwargs.get("tokenizer"):
                kwargs["tokenizer"].save_pretrained(output_checkpoint_dir)

def find_latest_checkpoint(path):
    """
    Trova l'ultimo checkpoint salvato per riprendere l'addestramento.
    """
    if not os.path.exists(path):
        return None
    
    checkpoints = [d for d in os.listdir(path) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    
    latest_checkpoint = max(checkpoints, key=lambda cp: int(cp.split('-')[1]))
    return os.path.join(path, latest_checkpoint)

# ==============================================================================
# SEZIONE 3: FUNZIONE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, progress_container):
    """
    Esegue il fine-tuning del modello di linguaggio sul corpus fornito.
    """
    try:
        # Step 1: Inizializza il tokenizer e il modello
        progress_container("Inizializzazione del tokenizer e del modello...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

        # Step 2: Prepara il dataset
        progress_container("Preparazione del dataset di addestramento...", "info")
        dataset = Dataset.from_pandas(corpus_df)
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Step 3: Configura PEFT (LoRA)
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

        # Step 4: Configura e avvia l'addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=1,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_strategy="steps",
            logging_steps=50,
            save_strategy="steps",
            save_steps=500,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            eval_steps=500
        )
        
        last_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            progress_container(f"Trovato l'ultimo checkpoint: {last_checkpoint}. Riprendo l'addestramento...", "info")
        else:
            progress_container("Nessun checkpoint trovato. Avvio un nuovo addestramento...", "info")
            
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
