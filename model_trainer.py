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
        """
        Controlla se è il momento di salvare il modello.
        """
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            output_dir_step = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(output_dir_step)
            kwargs['tokenizer'].save_pretrained(output_dir_step)
            print(f"Modello salvato al passo {state.global_step}")
            return control
        return control

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def train_model(corpus_df, progress_container):
    """
    Avvia il processo di fine-tuning del modello.
    """
    try:
        progress_container("Preparazione dei dati per l'addestramento...", "info")
        dataset = Dataset.from_pandas(corpus_df)
        dataset = dataset.train_test_split(test_size=0.1)
        train_dataset = dataset['train']
        eval_dataset = dataset['test']
        
        # Step 1: Carica il tokenizer e il modello base
        progress_container("Caricamento del modello base...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Aggiungi un token di fine-sequenza, se non è già presente
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '</s>'})
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        model.resize_token_embeddings(len(tokenizer))

        # Step 2: Configura PEFT (LoRA)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        peft_model = get_peft_model(model, lora_config)
        progress_container("Configurazione PEFT applicata con successo.", "success")
        peft_model.print_trainable_parameters()

        # Step 3: Configura gli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            learning_rate=2e-4,
            fp16=False,
            bf16=True,
            logging_steps=100,
            save_total_limit=2,
            push_to_hub=False,
        )

        # Cerca l'ultimo checkpoint
        last_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            subdirectories = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
            checkpoint_dirs = [d for d in subdirectories if d.startswith("checkpoint-")]
            if checkpoint_dirs:
                last_checkpoint = os.path.join(OUTPUT_DIR, sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[1]))[-1])
                progress_container(f"Trovato l'ultimo checkpoint: {last_checkpoint}. L'addestramento riprenderà da qui.", "info")
            
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
