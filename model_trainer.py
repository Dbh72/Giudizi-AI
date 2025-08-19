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
        """Salva il modello ogni N passi."""
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            kwargs["model"].save_pretrained(checkpoint_dir)
            kwargs["tokenizer"].save_pretrained(checkpoint_dir)
            
            # Pulisce i checkpoint vecchi, mantenendo solo l'ultimo
            for f in os.listdir(self.output_dir):
                if f.startswith("checkpoint-") and f != f"checkpoint-{state.global_step}":
                    shutil.rmtree(os.path.join(self.output_dir, f))

# ==============================================================================
# SEZIONE 3: LOGICA PRINCIPALE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, progress_container):
    """
    Esegue il fine-tuning del modello T5 con PEFT/LoRA.
    
    Args:
        corpus_df (pd.DataFrame): DataFrame contenente i dati di addestramento.
        progress_container (callable): Funzione per inviare messaggi di stato.
    
    Returns:
        tuple: (model, tokenizer) o (None, None) in caso di errore.
    """
    try:
        progress_container("Avvio del fine-tuning del modello...", "info")
        
        # Step 1: Prepara i dati
        progress_container("Preparazione dei dati per il fine-tuning...", "info")
        dataset = Dataset.from_pandas(corpus_df)
        
        # Carica il tokenizer
        progress_container("Caricamento del modello base e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Funzione di pre-processamento che tokenizza i dati
        def preprocess_function(examples):
            inputs = [f"generate judgment: {ex}" for ex in examples["input_text"]]
            targets = [ex for ex in examples["target_text"]]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)
            labels = tokenizer(targets, max_length=512, truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Applica il pre-processing al dataset
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input_text", "target_text"])
        tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = tokenized_dataset["train"]
        eval_dataset = tokenized_dataset["test"]
        
        # Carica il modello base
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")

        # Step 2: Configura PEFT/LoRA
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

        # Step 3: Configura e avvia il Trainer
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=1,
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            overwrite_output_dir=True,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            report_to="none", # Disabilita i report esterni
            remove_unused_columns=False, # Importante per evitare l'errore
        )
        
        # Cerca il checkpoint più recente per il resume
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            for d in os.listdir(OUTPUT_DIR):
                if d.startswith("checkpoint-"):
                    last_checkpoint = os.path.join(OUTPUT_DIR, d)
        
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
