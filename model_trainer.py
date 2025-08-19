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
        self.last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """
        Controlla se è il momento di salvare il checkpoint.
        """
        if state.global_step % self.save_steps == 0 and state.global_step > self.last_step:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            print(f"Salvataggio checkpoint a passo {state.global_step} in {checkpoint_dir}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            self.last_step = state.global_step
            
# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def fine_tune(corpus_df, progress_container):
    """
    Esegue il fine-tuning del modello.
    """
    try:
        progress_container("Inizio del processo di fine-tuning...", "info")
        
        # Step 1: Preparazione del dataset
        progress_container("Preparazione del dataset...", "info")
        if corpus_df.empty:
            progress_container("Errore: Il DataFrame del corpus è vuoto. Impossibile addestrare il modello.", "error")
            return None, None
            
        dataset = Dataset.from_pandas(corpus_df)
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        # Step 2: Caricamento del modello e del tokenizer
        progress_container(f"Caricamento del modello base '{MODEL_NAME}'...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

        # Step 3: Configurazione PEFT (LoRa) e TrainingArguments
        progress_container("Configurazione di PEFT (LoRa)...", "info")
        
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

        # Verifica la presenza di checkpoint precedenti
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
            if checkpoints:
                last_checkpoint_path = max([os.path.join(OUTPUT_DIR, d) for d in checkpoints], key=os.path.getmtime)
                last_checkpoint = last_checkpoint_path
                progress_container(f"Trovato un checkpoint precedente: {last_checkpoint}", "info")
            
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            evaluation_strategy="steps", # Correzione: `evaluation_strategy`
            load_best_model_at_end=True,
            report_to="none"
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
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
            progress_container(f"Directory '{OUTPUT_DIR}' rimossa a causa dell'errore.", "warning")
        return None, None

