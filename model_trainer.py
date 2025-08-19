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
    ogni N passi di addestramento. Questo è utile per riprendere
    l'addestramento in caso di interruzione.
    """
    def __init__(self, output_dir, save_steps=500):
        super().__init__()
        self.output_dir = output_dir
        self.save_steps = save_steps
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Viene chiamato alla fine di ogni passo di addestramento.
        """
        if state.global_step % self.save_steps == 0:
            output_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Salva il modello e il tokenizer
            kwargs["model"].save_pretrained(output_dir)
            kwargs["tokenizer"].save_pretrained(output_dir)
            
            # Scrivi un file di stato per tracciare l'ultimo passo salvato
            state_file = os.path.join(output_dir, "trainer_state.json")
            with open(state_file, "w") as f:
                json.dump({"global_step": state.global_step}, f)
            
            print(f"Salvato il checkpoint in {output_dir}")
            
class LossLoggingCallback(TrainerCallback):
    """
    Un callback personalizzato per registrare e stampare la loss di addestramento
    a intervalli regolari.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Viene chiamato quando l'oggetto Trainer registra un log.
        """
        if state.global_step > 0 and 'loss' in logs:
            loss = logs['loss']
            print(f"Passo: {state.global_step}, Loss: {loss:.4f}")
            
# ==============================================================================
# SEZIONE 3: FUNZIONE PRINCIPALE PER IL FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df, progress_container):
    """
    Esegue il fine-tuning del modello di linguaggio.

    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente i dati di addestramento.
        progress_container (callable): Funzione per inviare messaggi di stato all'interfaccia utente.

    Returns:
        tuple: Il modello e il tokenizer fine-tuned.
    """
    try:
        if corpus_df.empty:
            progress_container("Errore: Il corpus di addestramento è vuoto. Impossibile avviare il fine-tuning.", "error")
            return None, None
            
        progress_container("Preparazione dei dati per il fine-tuning...", "info")
        
        # Step 1: Prepara il tokenizer e i dati
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def preprocess_function(examples):
            """Prepara gli input e i target per il modello."""
            inputs = [ex for ex in examples['input_text']]
            targets = [ex for ex in examples['target_text']]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)
            labels = tokenizer(text_target=targets, max_length=512, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = Dataset.from_pandas(corpus_df)
        tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=corpus_df.columns)
        
        # Suddividi il dataset in addestramento e valutazione
        dataset_dict = tokenized_dataset.train_test_split(test_size=0.1)
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['test']
        
        progress_container(f"Dataset di addestramento: {len(train_dataset)} esempi. Dataset di valutazione: {len(eval_dataset)} esempi.", "success")
        
        # Step 2: Carica il modello e configura PEFT
        progress_container("Caricamento del modello base e configurazione di PEFT...", "info")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME, 
            load_in_8bit=False,  # Set a False
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        peft_model = get_peft_model(base_model, lora_config)
        peft_model.print_trainable_parameters()
        
        # Step 3: Configura gli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=50,
            logging_steps=100,
            save_steps=500,
            save_total_limit=3,
            evaluation_strategy="steps",
            eval_steps=500,
            report_to="none",
            fp16=False,
            bf16=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            remove_unused_columns=False,
            dataloader_num_workers=2,
            gradient_checkpointing=True,
            gradient_accumulation_steps=4,
        )
        
        # Trova l'ultimo checkpoint per riprendere l'addestramento
        last_checkpoint = None
        if os.path.isdir(OUTPUT_DIR):
            dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
            checkpoints = [d for d in dirs if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint = os.path.join(OUTPUT_DIR, max(checkpoints, key=lambda d: int(d.split('-')[1])))
                progress_container(f"Trovato l'ultimo checkpoint: {last_checkpoint}. Riprenderò l'addestramento da qui.", "info")
        
        # Step 4: Avvia il Trainer
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
        # Rimuove la directory del modello se l'addestramento fallisce
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        return None, None
