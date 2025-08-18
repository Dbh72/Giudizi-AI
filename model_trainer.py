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

# Ignoriamo i FutureWarning per una console più pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE
# ==============================================================================

# Directory dove il modello addestrato verrà salvato.
OUTPUT_DIR = "modello_finetunato"
# Il nome del modello pre-addestrato di base da utilizzare.
MODEL_NAME = "google/flan-t5-base"

# ==============================================================================
# SEZIONE 3: CLASSI E CALLBACK PERSONALIZZATI
# ==============================================================================

class SaveEveryNStepsCallback(TrainerCallback):
    """
    Callback personalizzato per salvare il modello e lo stato ogni N step,
    consentendo la ripresa dell'addestramento.
    """
    def __init__(self, output_dir, save_steps=500):
        self.output_dir = output_dir
        self.save_steps = save_steps
        # Controlla se una sessione è già stata ripristinata
        self.has_been_resumed = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            output_path = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            print(f"Salvataggio del checkpoint in {output_path}")
            kwargs['model'].save_pretrained(output_path)
            kwargs['tokenizer'].save_pretrained(output_path)
            state.save_to_json(os.path.join(output_path, "trainer_state.json"))
            
            # Non rimuovere i checkpoint, servono per la ripresa
            # L'addestramento incrementale deve gestirla l'app
            
    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_main_process:
            self.has_been_resumed = state.global_step > 0
            if self.has_been_resumed:
                print(f"Ripresa dell'addestramento dal checkpoint: {state.global_step}")

# ==============================================================================
# SEZIONE 4: FUNZIONI PRINCIPALI
# ==============================================================================

def fine_tune_model(corpus_df, output_dir, num_epochs, learning_rate, batch_size, progress_container):
    """
    Funzione principale per l'addestramento del modello.
    
    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente il corpus di addestramento.
        output_dir (str): La directory dove salvare il modello.
        num_epochs (int): Il numero di epoche per l'addestramento.
        learning_rate (float): Il learning rate.
        batch_size (int): La dimensione del batch.
        progress_container (list): Una lista per i messaggi di progresso.
        
    Returns:
        str: Il percorso della directory del modello finale.
    """
    progress_container.append("Avvio del fine-tuning. Caricamento del tokenizer e del modello base...")
    try:
        # Step 1: Caricamento del tokenizer e del modello pre-addestrato
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Imposta `padding_side` a 'right' per i modelli Seq2Seq
        tokenizer.padding_side = "right"
        peft_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        progress_container.append("Tokenizer e modello base caricati con successo.")

        # Step 2: Preparazione del dataset e tokenizzazione
        progress_container.append("Conversione del DataFrame in un dataset e tokenizzazione...")
        dataset = Dataset.from_pandas(corpus_df)

        def tokenize_function(examples):
            # Tokenizzazione dei testi di input e target. La logica di chunking è gestita
            # dal Trainer in modo automatico se si usa una strategia appropriata.
            # Qui ci assicuriamo che i dati siano formattati correttamente per il training.
            model_inputs = tokenizer(
                examples['input_text'],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = tokenizer(
                examples['target_text'],
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            # Sostituisce i token di padding con -100 per l'addestramento
            labels['input_ids'][labels['input_ids'] == tokenizer.pad_token_id] = -100
            model_inputs['labels'] = labels['input_ids']
            return model_inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        # Suddivisione del dataset in training e validation
        train_test_split = tokenized_datasets.train_test_split(test_size=0.1)
        train_dataset = train_test_split['train']
        eval_dataset = train_test_split['test']
        
        progress_container.append("Dataset tokenizzato e suddiviso in training/validation.")

        # Step 3: Configurazione PEFT (Parameter-Efficient Fine-Tuning)
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        peft_model = get_peft_model(peft_model, peft_config)
        progress_container.append("Modello configurato con PEFT (LoRA).")

        # Verifica se un addestramento precedente può essere ripreso
        last_checkpoint = None
        if os.path.isdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith('checkpoint')]
            if checkpoints:
                last_checkpoint = os.path.join(output_dir, max(checkpoints, key=lambda d: int(d.split('-')[1])))
                progress_container.append(f"Trovato un checkpoint precedente: {last_checkpoint}. L'addestramento riprenderà da qui.")

        # Configurazione degli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="steps",
            save_steps=500,  # Salva ogni 500 step
            evaluation_strategy="steps", # Valuta ogni N passi
            eval_steps=500, # Valuta ogni 500 step
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            load_best_model_at_end=True, # Carica il modello migliore dopo l'addestramento
            report_to="none" # Disabilita i report per evitare dipendenze aggiuntive
        )

        # Step 4: Avvia il trainer
        progress_container.append("Avvio del processo di addestramento...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset, # Passa il set di validazione
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[SaveEveryNStepsCallback(output_dir=output_dir, save_steps=500)]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Step 5: Salva il modello finale
        progress_container.append("Addestramento completato. Salvataggio del modello finale...")
        final_model_path = os.path.join(output_dir, "final_model")
        peft_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        trainer.save_state()
        
        progress_container.append(f"Modello e tokenizer finali salvati in: {final_model_path}")
        return final_model_path

    except Exception as e:
        progress_container.append(f"Errore critico durante l'addestramento: {e}")
        progress_container.append(traceback.format_exc())
        print(f"Errore critico durante l'addestramento: {e}\n{traceback.format_exc()}")
        return None
