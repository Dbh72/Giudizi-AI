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
        """
        Registra la loss ogni 100 passi di addestramento.
        """
        # Controlliamo se stiamo addestrando e se il passo corrente è un multiplo di 100
        if state.global_step % 100 == 0 and state.global_step > 0:
            loss = kwargs["logs"]["loss"]
            print(f"Passo {state.global_step}: Loss = {loss}")
            
# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI DEL MODULO
# ==============================================================================

def tokenize_and_prepare_dataset(dataset, tokenizer):
    """
    Tokenizza e prepara il dataset per l'addestramento, creando prompt e target.
    
    Args:
        dataset (Dataset): Il dataset da preparare.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        
    Returns:
        Dataset: Il dataset tokenizzato e pronto per l'addestramento.
    """
    def preprocess_function(examples):
        """
        Funzione per processare ogni esempio nel dataset.
        Crea il prompt e tokenizza input e target.
        """
        # Creiamo un prompt che guida il modello a generare il giudizio
        # Inseriamo un controllo per evitare errori con dati vuoti o non validi
        prompts = []
        targets = []
        for i in range(len(examples['input_text'])):
            # Controlliamo che i dati non siano vuoti o di un tipo non valido
            if isinstance(examples['input_text'][i], str) and examples['input_text'][i].strip() and \
               isinstance(examples['target_text'][i], str) and examples['target_text'][i].strip():
                prompts.append(f"Genera un giudizio per la seguente valutazione: {examples['input_text'][i]}")
                targets.append(examples['target_text'][i])

        # Tokenizziamo i prompt e i target per renderli comprensibili al modello
        model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

        # Assegniamo i labels all'input del modello, essenziale per l'addestramento supervisionato
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
    
    # Applichiamo la funzione di preprocessing al dataset in batch per efficienza
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['input_text', 'target_text'])
    
    return tokenized_dataset

def train_model(corpus_df, progress_container):
    """
    Avvia l'addestramento del modello sul corpus di addestramento.
    
    Args:
        corpus_df (pd.DataFrame): Il DataFrame contenente il corpus.
        progress_container (callable): Funzione per inviare messaggi di stato.
        
    Returns:
        tuple: Il modello e il tokenizer addestrati, altrimenti (None, None).
    """
    try:
        # Verifica se il DataFrame è vuoto
        if corpus_df.empty:
            progress_container("Il corpus di addestramento è vuoto. Impossibile addestrare il modello.", "error")
            return None, None
            
        progress_container("Preparazione dei dati per l'addestramento...", "info")
        
        # Convertiamo il DataFrame in un oggetto Dataset di Hugging Face
        dataset = Dataset.from_pandas(corpus_df)
        
        # Suddividiamo il dataset in training e validation set (90% training, 10% validation)
        # Questo ci aiuta a valutare le performance del modello su dati non visti
        dataset_split = dataset.train_test_split(test_size=0.1)
        
        progress_container("Caricamento del tokenizer e del modello base...", "info")
        
        # Carichiamo il tokenizer e il modello pre-addestrato dal modello base (T5)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Utilizziamo torch.float32 per compatibilità con l'addestramento
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

        # Configurazione LoRA (Low-Rank Adaptation) per un fine-tuning efficiente
        # Questa tecnica riduce drasticamente il numero di parametri da addestrare
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )

        # Applichiamo la configurazione PEFT al modello base
        model = get_peft_model(base_model, peft_config)
        
        # Stampiamo un riepilogo dei parametri addestrabili per debug
        model.print_trainable_parameters()
        
        progress_container("Tokenizzazione del dataset...", "info")
        # Applichiamo la funzione di preparazione e tokenizzazione al dataset
        tokenized_datasets = tokenize_and_prepare_dataset(dataset_split, tokenizer)
        
        # Configurazione degli argomenti per l'addestramento (TrainingArguments)
        # Questi parametri controllano il processo di fine-tuning
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100,
            report_to="none",
            remove_unused_columns=False,
        )
        
        # Data Collator per l'imbottitura delle sequenze
        # Si occupa di "imbottire" le sequenze in modo che abbiano tutte la stessa lunghezza
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding="max_length"
        )
        
        # Inizializzazione del Trainer di Hugging Face
        # Il Trainer gestisce l'intero ciclo di addestramento
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            data_collator=data_collator,
            callbacks=[LossLoggingCallback()],
        )
        
        progress_container("Avvio del fine-tuning...", "info")
        # Avviamo il processo di addestramento
        trainer.train()
        
        progress_container("Addestramento completato. Salvataggio del modello...", "info")
        
        # Salviamo il modello e il tokenizer nella directory di output
        final_model_path = os.path.join(OUTPUT_DIR, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        progress_container("Modello salvato con successo.", "success")
        return model, tokenizer
        
    except Exception as e:
        progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
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
        try:
            shutil.rmtree(OUTPUT_DIR)
            progress_container("Modello fine-tuned eliminato con successo.", "success")
        except OSError as e:
            progress_container(f"Errore: {e.filename} - {e.strerror}.", "error")
    else:
        progress_container("Nessun modello da eliminare.", "warning")
