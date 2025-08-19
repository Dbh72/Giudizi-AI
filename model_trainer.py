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
        """
        Controlla se è il momento di salvare il modello.
        """
        if state.global_step % self.save_steps == 0 and state.global_step > self.last_saved_step:
            save_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(save_path)
            kwargs['tokenizer'].save_pretrained(save_path)
            self.last_saved_step = state.global_step
            print(f"Modello salvato al checkpoint: {save_path}")

class LossLoggingCallback(TrainerCallback):
    """
    Un callback per loggare la loss durante l'addestramento,
    utile per il monitoraggio.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None and "loss" in logs:
            print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def train_model(train_dataset: Dataset, eval_dataset: Dataset, progress_container):
    """
    Esegue il fine-tuning del modello T5.
    
    Argomenti:
        train_dataset (Dataset): Il set di dati di addestramento.
        eval_dataset (Dataset): Il set di dati di valutazione.
        progress_container (function): Funzione per aggiornare lo stato dell'interfaccia utente.
    """
    try:
        # Step 1: Caricamento del modello e del tokenizer
        progress_container("Caricamento del modello e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token="hf_jOQxNlJpQdJbZlJ...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token="hf_jOQxNlJpQdJbZlJ...",
                                                     device_map="auto")

        # Step 2: Configurazione di LoRA
        progress_container("Configurazione di LoRA...", "info")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v", "k"],  # Moduli del modello su cui applicare LoRA
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False
        )
        
        # Applica LoRA al modello di base
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Step 3: Configurazione degli argomenti di addestramento
        # NOTA: Qui ho rimosso 'evaluation_strategy' e 'eval_steps' in quanto non più supportati
        # da transformers 4.55.2.
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=50,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=50,
            weight_decay=0.01,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            report_to=["none"], # Disabilita i report per evitare errori con W&B, TensorBoard, ecc.
            seed=42,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
        )
        
        # Step 4: Avvio del Trainer
        # Controlla se esistono checkpoint precedenti per riprendere l'addestramento
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
            for d in reversed(sorted(dirs)):
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
