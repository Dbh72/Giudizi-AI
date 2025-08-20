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
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            if state.is_local_process_zero:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Passo {state.global_step}: Loss = {state.log_history[-1]['loss']:.4f}")

class SaveModelCallback(TrainerCallback):
    """
    Callback per salvare il modello in modo incrementale a intervalli specifici.
    """
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % 500 == 0:
            output_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(output_dir)
            if state.is_local_process_zero:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Checkpoint salvato al passo {state.global_step}")

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI PER L'ADDESTRAMENTO
# ==============================================================================

def train_model(corpus_df, progress_container):
    """
    Prepara e avvia il processo di fine-tuning del modello.
    
    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente il corpus di addestramento.
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
        
    Returns:
        tuple: Una tupla contenente il modello e il tokenizer addestrati, altrimenti (None, None).
    """
    try:
        progress_container("Inizio del processo di addestramento...", "info")
        
        # 1. Caricamento del modello e del tokenizer pre-addestrati
        progress_container(f"Caricamento di `{MODEL_NAME}` e del tokenizer...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Aggiungi un token per la fine del testo se non esiste
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})

        # 2. Preparazione dei dati per il dataset
        progress_container("Preparazione del dataset...", "info")
        def preprocess_function(examples):
            # Formattazione per il modello text-to-text
            inputs = [f"{ex['input_text']}" for ex in examples]
            model_inputs = tokenizer(inputs, max_length=512, truncation=True)

            # Setup per il tokenizer target
            labels = tokenizer(text_target=[ex['target_text'] for ex in examples], max_length=512, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        # Crea un Dataset da un DataFrame
        dataset = Dataset.from_pandas(corpus_df)
        tokenized_dataset = dataset.map(preprocess_function, batched=True)

        # 3. Configurazione PEFT (Parameter-Efficient Fine-Tuning)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 4. Argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=3,
            logging_dir=os.path.join(OUTPUT_DIR, "logs"),
            logging_strategy="steps",
            logging_steps=100,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=False,
        )

        # 5. Data collator per il padding
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # 6. Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[SaveModelCallback(), LossLoggingCallback()]
        )

        # 7. Avvio dell'addestramento
        trainer.train()

        # 8. Salvataggio del modello finale
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        if not os.path.exists(final_model_dir):
            os.makedirs(final_model_dir)

        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        progress_container("Modello finale e tokenizer salvati con successo.", "success")
        
        return model, tokenizer

    except Exception as e:
        progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

def load_fine_tuned_model(progress_container):
    """
    Carica un modello fine-tuned e il suo tokenizer da una directory.
    
    Args:
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
        
    Returns:
        tuple: Una tupla contenente il modello e il tokenizer caricati, altrimenti (None, None).
    """
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        if not os.path.exists(model_path):
            progress_container("Nessun modello fine-tuned trovato. Verrà caricato il modello pre-addestrato di base.", "warning")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            return model, tokenizer

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
    
    Args:
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        progress_container("Modello fine-tuned eliminato.", "success")
    else:
        progress_container("Nessun modello da eliminare.", "warning")
