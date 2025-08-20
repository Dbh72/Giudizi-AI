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
            print(f"Passo {state.global_step}: Loss = {state.global_step_loss}")


# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def fine_tune_model(corpus_df, progress_container, training_args=None):
    """
    Esegue il fine-tuning del modello di linguaggio.

    Args:
        corpus_df (pd.DataFrame): DataFrame contenente i dati di addestramento.
        progress_container (callable): Funzione per inviare messaggi di stato a Streamlit.
        training_args (TrainingArguments, optional): Argomenti di addestramento.
                                                    Se None, vengono usati i valori di default.

    Returns:
        tuple: Il modello e il tokenizer fine-tunati.
    """
    try:
        if corpus_df.empty:
            progress_container("Il corpus è vuoto. Impossibile avviare l'addestramento.", "error")
            return None, None

        progress_container("Avvio del processo di fine-tuning...", "info")

        # Step 1: Carica il modello e il tokenizer
        progress_container(f"Caricamento del modello base: {MODEL_NAME}...", "info")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        # Assicuriamoci che il tokenizer abbia un pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            progress_container("Aggiunto pad_token al tokenizer.", "info")

        # Step 2: Prepara il dataset per l'addestramento
        progress_container("Preparazione del dataset...", "info")
        dataset = Dataset.from_pandas(corpus_df)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        # Step 3: Configura il modello con PEFT (LoRA)
        progress_container("Configurazione del modello con PEFT...", "info")
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

        # Step 4: Configura e avvia il Trainer
        progress_container("Configurazione del Trainer...", "info")
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                learning_rate=3e-4,
                num_train_epochs=10,
                evaluation_strategy="steps",
                eval_steps=500,
                logging_steps=100,
                save_steps=500,
                save_total_limit=3,
                load_best_model_at_end=True,
                report_to="none",  # Disabilita i report per semplicità
                bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 # Abilita bf16 se disponibile
            )

        # Verifica se esiste un checkpoint precedente e riprendi da lì
        last_checkpoint = None
        if os.path.exists(OUTPUT_DIR):
            dirs = os.listdir(OUTPUT_DIR)
            for d in sorted(dirs, reverse=True):
                if d.startswith("checkpoint-"):
                    last_checkpoint = os.path.join(OUTPUT_DIR, d)
                    progress_container(f"Riprendendo l'addestramento dall'ultimo checkpoint: {last_checkpoint}", "info")
                    break
        
        progress_container("Avvio del processo di addestramento...", "info")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[LossLoggingCallback()] # Abbiamo rimosso il callback personalizzato per il salvataggio
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)
        
        # Step 5: Salva il modello finale
        progress_container("Addestramento completato. Salvataggio del modello finale...", "info")
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        
        # Salva il modello e il tokenizer
        # Il Trainer di Hugging Face gestisce già il salvataggio corretto di modello e tokenizer
        trainer.save_model(final_model_path)
        
        # Salva lo stato del trainer
        trainer.save_state()
        
        progress_container(f"Modello e tokenizer finali salvati in: {final_model_path}", "success")
        return peft_model, tokenizer
    
    except Exception as e:
        progress_container(f"Errore durante l'addestramento: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

def load_fine_tuned_model(progress_container):
    """
    Carica un modello fine-tuned e il suo tokenizer da una directory.
    """
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
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
    """
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        progress_container("Modello addestrato eliminato.", "success")
    else:
        progress_container("Nessun modello da eliminare.", "warning")

