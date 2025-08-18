# ==============================================================================
# File: model_trainer.py
# Modulo per la logica di addestramento (fine-tuning) del modello.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import os
import streamlit as st
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from transformers.trainer_callback import TrainerCallback
import torch
import shutil
import warnings

# Ignoriamo i FutureWarning per mantenere la console pulita
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE
# ==============================================================================
OUTPUT_DIR = "modello_finetunato"
MODEL_NAME = "google/flan-t5-base"

# ==============================================================================
# SEZIONE 3: CLASSE DI CALLBACK PER STREAMLIT
# ==============================================================================
class StreamlitCallback(TrainerCallback):
    """
    Una callback personalizzata per aggiornare lo stato dell'interfaccia utente di Streamlit
    durante l'addestramento del modello.
    """
    def __init__(self, add_status_message):
        self.add_status_message = add_status_message
        self.total_steps = 0
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Chiamato all'inizio dell'addestramento."""
        self.total_steps = state.max_steps
        self.add_status_message(f"Addestramento avviato per {self.total_steps} step.")
        # Usiamo un'istruzione condizionale per evitare errori di st.progress
        if 'progress_bar' not in st.session_state:
            st.session_state.progress_bar = st.progress(0, text="Progresso addestramento...")
        self.progress_bar = st.session_state.progress_bar

    def on_step_end(self, args, state, control, **kwargs):
        """Chiamato alla fine di ogni step di addestramento."""
        if self.progress_bar:
            progress_percentage = state.global_step / self.total_steps
            self.progress_bar.progress(progress_percentage, text=f"Step {state.global_step}/{self.total_steps} completato.")
        
        # Salvataggio automatico del modello ogni 500 step
        if state.global_step % 500 == 0:
            control.should_save = True
            self.add_status_message(f"Checkpoint logico: Salvataggio del modello allo step {state.global_step}.")
        
        # Esegui la valutazione ogni 500 step
        if state.global_step % 500 == 0 and state.global_step > 0:
            control.should_evaluate = True
            self.add_status_message(f"Esecuzione della valutazione allo step {state.global_step}.")

    def on_save(self, args, state, control, **kwargs):
        """Chiamato ogni volta che il modello viene salvato."""
        self.add_status_message(f"Modello salvato in: {control.save_path}")

    def on_train_end(self, args, state, control, **kwargs):
        """Chiamato alla fine dell'addestramento."""
        if self.progress_bar:
            self.progress_bar.progress(1.0, text="Addestramento completato!")
        self.add_status_message("Fine-tuning terminato con successo.")

# ==============================================================================
# SEZIONE 4: FUNZIONI PRINCIPALI
# ==============================================================================

def fine_tune_model(corpus_df):
    """
    Esegue il fine-tuning di un modello T5-base con i dati forniti.

    Args:
        corpus_df (pd.DataFrame): DataFrame contenente le colonne 'input_text' e 'target_text'.

    Returns:
        str: Il percorso della directory del modello salvato, o None in caso di errore.
    """
    try:
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Inizializza tokenizer e modello
        add_status_message = st.session_state.add_status_message
        
        add_status_message(f"Caricamento tokenizer e modello base '{MODEL_NAME}'...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Configurazione PEFT/LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)
        
        add_status_message("Modello configurato per il fine-tuning LoRA.")
        
        # Prepara il dataset di Hugging Face
        add_status_message("Conversione del DataFrame in Dataset Hugging Face...")
        from datasets import Dataset
        dataset = Dataset.from_pandas(corpus_df)
        
        def tokenize_function(examples):
            tokenized_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True)
            tokenized_labels = tokenizer(examples['target_text'], max_length=512, truncation=True)
            tokenized_inputs["labels"] = tokenized_labels["input_ids"]
            return tokenized_inputs

        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        
        add_status_message("Dataset tokenizzato. Pronto per l'addestramento.")

        # Configurazione degli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{OUTPUT_DIR}/logs',
            logging_steps=100,
            save_steps=500,
            evaluation_strategy="steps", # Allineato con save_steps
            eval_steps=500,
            load_best_model_at_end=True,
            report_to="none"
        )
        
        # Inizializzazione del Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer),
            callbacks=[StreamlitCallback(add_status_message)]
        )
        
        # Avvio dell'addestramento
        trainer.train()
        
        # Salvataggio finale del modello
        final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
        
        add_status_message(f"Modello finale salvato in: {final_model_dir}")
        return final_model_dir

    except Exception as e:
        print(f"Errore nel fine-tuning: {e}")
        return None
