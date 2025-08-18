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
    Un callback personalizzato per salvare il modello ogni N passi di addestramento.
    Questo crea dei checkpoint riutilizzabili.
    """
    def __init__(self, output_dir, save_steps=500):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_saved_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > self.last_saved_step:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            if kwargs['tokenizer']:
                kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            print(f"Checkpoint salvato a {state.global_step} passi in {checkpoint_dir}")
            self.last_saved_step = state.global_step

# ==============================================================================
# SEZIONE 4: FUNZIONI PRINCIPALI DI PRE-PROCESSING E FINE-TUNING
# ==============================================================================

def _chunk_and_tokenize_dataset(tokenizer, raw_dataset):
    """
    Processa il dataset applicando la tokenizzazione e la logica di chunking per
    gestire i testi che superano la lunghezza massima del modello.
    """
    MAX_LENGTH = 512
    CHUNK_OVERLAP = 50

    processed_data = []

    for item in raw_dataset:
        input_text = item['input_text']
        target_text = item['target_text']
        
        # Tokenizza il testo di input.
        input_tokens = tokenizer(
            input_text,
            max_length=MAX_LENGTH,
            truncation=True,
            return_overflowing_tokens=True,
            stride=CHUNK_OVERLAP,
            padding="max_length"
        )
        
        # Se il testo è troppo lungo, lo spezziamo in chunk.
        if "overflowing_tokens" in input_tokens and len(input_tokens["overflowing_tokens"]) > 0:
            print(f"Testo troppo lungo, suddivisione in {len(input_tokens['input_ids'])} chunk...")
            for chunk_id in input_tokens["input_ids"]:
                chunk_text = tokenizer.decode(chunk_id, skip_special_tokens=True)
                processed_data.append({
                    'input_text': chunk_text,
                    'target_text': target_text  # Il target rimane lo stesso per tutti i chunk
                })
        else:
            processed_data.append(item)
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))


def fine_tune_model(progress_container, fine_tune_file, output_dir=OUTPUT_DIR):
    """
    Esegue il fine-tuning del modello.
    Aggiunge la gestione dello stato e i messaggi di avanzamento.
    """
    progress_container.append("Avvio del processo di fine-tuning...")
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Carica il dataset dalla memoria
        raw_df = pd.read_excel(fine_tune_file.name)
        
        # Converti il DataFrame in un Dataset Hugging Face
        # Supponiamo che il DataFrame abbia colonne 'input_text' e 'target_text'
        # Dobbiamo prima assicurarci che il DataFrame non sia vuoto
        if raw_df.empty:
            progress_container.append("Errore: Il DataFrame caricato è vuoto.")
            return None
        
        raw_dataset = Dataset.from_pandas(raw_df)

        progress_container.append("Caricamento e preparazione del tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Applica il chunking e la tokenizzazione
        progress_container.append("Applicazione della logica di chunking e tokenizzazione...")
        processed_dataset = _chunk_and_tokenize_dataset(tokenizer, raw_dataset)

        # Suddividi il dataset in training e validation set
        progress_container.append("Suddivisione del dataset in set di addestramento e validazione...")
        split_datasets = processed_dataset.train_test_split(test_size=0.1)
        train_dataset = split_datasets['train']
        eval_dataset = split_datasets['test']

        progress_container.append(f"Set di addestramento: {len(train_dataset)} esempi. Set di validazione: {len(eval_dataset)} esempi.")
        
        progress_container.append("Caricamento del modello di base...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        progress_container.append("Configurazione del fine-tuning con PEFT (LoRa)...")
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

        # Verifica se esiste un checkpoint precedente per riprendere l'addestramento
        last_checkpoint = None
        if os.path.exists(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Trova il checkpoint con il numero di step più alto
                checkpoints.sort(key=lambda x: int(x.split('-')[1]))
                last_checkpoint = os.path.join(output_dir, checkpoints[-1])
                progress_container.append(f"Trovato un checkpoint precedente: {last_checkpoint}. L'addestramento riprenderà da qui.")

        progress_container.append("Definizione degli argomenti per il training...")
        training_args = TrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=3,
            per_device_train_batch_size=2, # Ridotto per stabilità, ma può essere adattato
            per_device_eval_batch_size=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_steps=500, # Salva ogni 500 step
            evaluation_strategy="steps",
            eval_steps=500, # Valuta ogni 500 step
            load_best_model_at_end=True,
        )

        progress_container.append("Avvio del processo di addestramento...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=peft_model),
            callbacks=[SaveEveryNStepsCallback(output_dir=output_dir, save_steps=500)]
        )

        trainer.train(resume_from_checkpoint=last_checkpoint)

        progress_container.append("Addestramento completato. Salvataggio del modello finale...")
        final_model_path = os.path.join(output_dir, "final_model")
        peft_model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        trainer.save_state()
        
        progress_container.append(f"Modello e tokenizer finali salvati in: {final_model_path}")
        return final_model_path

    except Exception as e:
        error_message = f"Errore durante il fine-tuning: {e}\n\nTraceback:\n{traceback.format_exc()}"
        progress_container.append(error_message)
        print(error_message)
        return None

