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
    """
    def __init__(self, output_dir, save_steps):
        self.output_dir = output_dir
        self.save_steps = save_steps
        self.last_save_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        """
        Salva il modello alla fine di ogni passo specificato.
        """
        if state.global_step > self.last_save_step and state.global_step % self.save_steps == 0:
            print(f"Salvataggio del checkpoint al passo {state.global_step}...")
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
            kwargs['model'].save_pretrained(checkpoint_dir)
            kwargs['tokenizer'].save_pretrained(checkpoint_dir)
            print(f"Checkpoint salvato in: {checkpoint_dir}")
            self.last_save_step = state.global_step

# ==============================================================================
# SEZIONE 4: FUNZIONI PRINCIPALI
# ==============================================================================

def train_model(df_train, progress_container, output_dir):
    """
    Esegue il fine-tuning del modello T5.
    
    Args:
        df_train (pd.DataFrame): DataFrame contenente i dati di addestramento.
        progress_container (list): Lista per i messaggi di stato di Streamlit.
        output_dir (str): La directory dove salvare i file del modello.
    
    Returns:
        str: Il percorso del modello salvato.
    """
    try:
        # Step 1: Carica tokenizer e modello base
        progress_container.append("Caricamento del tokenizer e del modello base...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Step 2: Preparazione dei dati e creazione del dataset
        progress_container.append("Preparazione del dataset...")
        
        # Assicurarsi che le colonne abbiano i nomi corretti
        if 'input_text' not in df_train.columns or 'target_text' not in df_train.columns:
            raise ValueError("Il DataFrame di addestramento deve avere le colonne 'input_text' e 'target_text'.")
        
        train_dataset = Dataset.from_pandas(df_train).map(
            lambda examples: tokenizer(examples['input_text'], truncation=True, padding='max_length', max_length=512),
            batched=True,
            remove_columns=['input_text', 'target_text']
        )
        train_dataset = train_dataset.map(
            lambda examples: tokenizer(examples['target_text'], truncation=True, padding='max_length', max_length=150),
            batched=True,
            rename_columns={'input_ids': 'input_ids', 'attention_mask': 'attention_mask', 'labels': 'labels'}
        )
        
        # Aggiungo un dataset di valutazione, ad esempio 10% del dataset
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'eval': train_dataset.select(range(int(len(train_dataset) * 0.1))) # Semplice divisione per esempio
        })
        
        train_dataset = dataset_dict['train']
        eval_dataset = dataset_dict['eval']

        # Step 3: Configurazione di PEFT (Parameter-Efficient Fine-Tuning)
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

        # Cerca l'ultimo checkpoint per la ripresa
        last_checkpoint = None
        if os.path.exists(output_dir) and os.listdir(output_dir):
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                last_checkpoint_path = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
                last_checkpoint = last_checkpoint_path
                progress_container.append(f"Trovato l'ultimo checkpoint: {last_checkpoint}")

        # Configurazione degli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_steps=500,
            evaluation_strategy="steps",
            eval_steps=500,
            load_best_model_at_end=True,
            report_to="none"
        )

        # Step 4: Avvia il trainer
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
        progress_container.append(f"Traceback:\n{traceback.format_exc()}")
        return None
