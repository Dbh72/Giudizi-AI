# ==============================================================================
# File: model_trainer.py
# Modulo per l'addestramento (Fine-Tuning) del modello di linguaggio.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo tutte le librerie essenziali per l'addestramento.
import torch
import warnings
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os
import shutil
import traceback
from datetime import datetime
from transformers.trainer_callback import TrainerCallback
import json

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
    Un callback personalizzato per salvare il modello e lo stato del trainer
    ogni N passi di addestramento.
    """
    def __init__(self, save_steps=500, output_dir="."):
        self.save_steps = save_steps
        self.output_dir = output_dir

    def on_step_end(self, args, state, control, **kwargs):
        """
        Viene chiamata alla fine di ogni passo di addestramento.
        """
        if state.global_step > 0 and state.global_step % self.save_steps == 0:
            print(f"Checkpoint! Salvataggio del modello al passo {state.global_step}...")
            # Salva il modello PEFT (solo gli adattatori LoRA)
            model = kwargs.get("model")
            if model:
                model.save_pretrained(os.path.join(self.output_dir, f"checkpoint-{state.global_step}"))
            
            # Salva lo stato del trainer per la resumibilità
            trainer = kwargs.get("trainer")
            if trainer:
                trainer.save_state()
            
            print(f"Modello e stato del trainer salvati in {self.output_dir}/checkpoint-{state.global_step}")

# ==============================================================================
# SEZIONE 4: FUNZIONE PRINCIPALE DI FINE-TUNING
# ==============================================================================

def fine_tune_model(corpus_df):
    """
    Esegue il fine-tuning del modello T5 con un dataset fornito.

    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente le colonne 'input_text' e 'target_text'.

    Returns:
        str: Il percorso della directory dove il modello addestrato è stato salvato.
    """
    try:
        # Step 1: Prepara il dataset e il tokenizer
        print("Preparazione del dataset e del tokenizer...")
        dataset = Dataset.from_pandas(corpus_df)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        def preprocess_function(examples):
            """
            Funzione per tokenizzare gli esempi di input e target.
            """
            model_inputs = tokenizer(examples['input_text'], max_length=512, truncation=True, padding="max_length")
            labels = tokenizer(examples['target_text'], max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_dataset = dataset.map(preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)

        # Step 2: Carica il modello e configura LoRA
        print("Caricamento del modello e configurazione PEFT (LoRA)...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        # Implementazione di LoRA per un fine-tuning più efficiente
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Step 3: Configura gli argomenti di addestramento
        print("Configurazione degli argomenti di addestramento...")
        # Aggiungiamo un timestamp per rendere la directory di output unica
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_output_dir = os.path.join(OUTPUT_DIR, f"model_{timestamp}")

        training_args = TrainingArguments(
            output_dir=final_output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=3,
            logging_dir=f'{final_output_dir}/logs',
            logging_steps=100,
            save_steps=500, # Salvataggio ogni 500 step
            save_total_limit=2,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none", # Disabilita i report per non richiedere account esterni
            fp16=True, # Abilita l'uso del float a 16 bit per velocità e minor consumo di memoria (richiede GPU)
            push_to_hub=False # Non caricare il modello su Hugging Face Hub
        )

        # Step 4: Avvia il trainer
        print("Avvio del processo di addestramento...")
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[SaveEveryNStepsCallback(output_dir=final_output_dir)]
        )

        trainer.train()

        # Step 5: Salva il modello finale
        final_model_path = os.path.join(final_output_dir, "final_model")
        peft_model.save_pretrained(final_model_path)
        print(f"Addestramento completato. Modello salvato in: {final_model_path}")

        # Salva il tokenizer
        tokenizer.save_pretrained(final_model_path)
        print(f"Tokenizer salvato in: {final_model_path}")

        # Salva lo stato di addestramento
        trainer.save_state()
        print(f"Stato di addestramento salvato.")

        # Simuliamo il salvataggio di un file che verrà zippato
        os.makedirs("dummy_model_directory", exist_ok=True)
        with open("dummy_model_directory/model_info.json", "w") as f:
            json.dump({"model": MODEL_NAME, "epochs": training_args.num_train_epochs}, f)

        return final_model_path
    
    except Exception as e:
        print("Errore durante il fine-tuning del modello.")
        traceback.print_exc()
        return None

