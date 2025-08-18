# app.py
# Aggiornamento basato su 33 Funziona.txt per implementare le logiche del Riepilogo Avanzato.

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import gradio as gr
import torch
import shutil
import json
from datetime import datetime
import openpyxl
from transformers.trainer_callback import TrainerCallback
import traceback
import time

# Importiamo il modulo di lettura Excel aggiornato
from excel_reader import load_and_prepare_excel, chunk_text_with_overlap

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE E CARICAMENTO MODELLO
# ==============================================================================
# Directory dove il modello addestrato verrà salvato.
OUTPUT_DIR = "modello_finetunato"
# Il modello base che stiamo usando.
MODEL_NAME = "google/flan-t5-small"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Check per la disponibilità della GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# SEZIONE 3: GESTIONE DELLO STATO E LOGICA APPLICATIVA
# ==============================================================================
# Inizializzazione del tokenizer e del modello base
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def load_initial_model():
    """
    Carica il modello base e lo prepara per il fine-tuning.
    Se esiste un modello finetunato, lo carica e lo restituisce.
    """
    if os.path.exists(OUTPUT_DIR):
        print("Caricamento del modello finetunato esistente...")
        model = PeftModel.from_pretrained(model_base, OUTPUT_DIR)
        return model.to(DEVICE)
    else:
        print("Nessun modello finetunato trovato. Inizializzazione del modello base.")
        return model_base.to(DEVICE)

def fine_tune_model(fine_tuning_state, file_input, sheet_name):
    """
    Avvia il processo di fine-tuning del modello.
    Ora utilizza la nuova logica da excel_reader.py.
    """
    fine_tuning_state['status'] = 'running'
    fine_tuning_state['message'] = "Avvio del fine-tuning..."
    
    # Puliamo la directory di output se esiste
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    try:
        # Carica e prepara i dati usando la logica aggiornata
        fine_tuning_state['message'] = "Caricamento e preparazione dei dati..."
        df = load_and_prepare_excel(file_input.name, sheet_name)
        if df is None or df.empty:
            fine_tuning_state['status'] = 'error'
            fine_tuning_state['message'] = "Errore: Dati non trovati o file vuoto."
            return fine_tuning_state
        
        # Prepara il dataset Hugging Face
        dataset = Dataset.from_pandas(df)
        
        # Funzione per tokenizzare e fare chunking con overlap
        def preprocess_function(examples):
            # `chunk_text_with_overlap` restituisce una lista di input_ids
            model_inputs = chunk_text_with_overlap(examples['input_text'], tokenizer)
            labels = tokenizer(examples['target_text'], max_length=128, truncation=True)
            
            # Il `Trainer` si aspetta input_ids e labels in un formato specifico
            # Qui mappiamo input e target, gestendo il padding
            processed_examples = []
            for i in range(len(model_inputs['input_ids'])):
                processed_examples.append({
                    'input_ids': model_inputs['input_ids'][i],
                    'attention_mask': model_inputs['attention_mask'][i],
                    'labels': labels['input_ids'][0] if i == 0 else [-100] * len(labels['input_ids'][0])
                    # Il label viene applicato solo al primo chunk per ogni esempio originale
                })
            
            # Ritorna una lista di dizionari per la corretta gestione da parte di `from_pandas`
            return processed_examples[0]
            
        tokenized_dataset = dataset.map(preprocess_function, batched=False, remove_columns=dataset.column_names)

        # Configurazione LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        peft_model = get_peft_model(model_base, lora_config)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=50,
            save_strategy="epoch",
            # Aggiunto per checkpointing regolare come da appunti
            save_total_limit=1,
            load_best_model_at_end=True,
        )

        # Aggiungi una callback per checkpoint più frequenti
        class MyCheckpointCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step % 500 == 0:
                    model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-{state.global_step}"))
        
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model)
        
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[MyCheckpointCallback()]
        )
        
        # Avvia l'addestramento
        trainer.train()
        
        # Salva il modello finale
        peft_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        
        fine_tuning_state['status'] = 'success'
        fine_tuning_state['message'] = "Fine-tuning completato con successo! Il modello è pronto."
    
    except Exception as e:
        traceback.print_exc()
        fine_tuning_state['status'] = 'error'
        fine_tuning_state['message'] = f"Errore durante il fine-tuning: {str(e)}"
    
    return fine_tuning_state

# ==============================================================================
# SEZIONE 4: INTERFACCIA GRADIO
# ==============================================================================
with gr.Blocks(title="Generatore Giudizi AI") as demo:
    gr.Markdown("# Generatore di Giudizi con IA")
    gr.Markdown("Carica il tuo file Excel e avvia l'addestramento dell'IA per generare giudizi basati sui tuoi dati.")
    
    fine_tuning_state = gr.State(value={'status': 'idle', 'message': 'In attesa di un file...'})

    # Controlli per il fine-tuning
    with gr.Row():
        with gr.Column(scale=1):
            fine_tune_file_input = gr.File(label="Carica file Excel per l'addestramento", file_types=[".xlsx", ".xls", ".xlsm"])
        with gr.Column(scale=1):
            fine_tune_sheet_dropdown = gr.Dropdown(label="Seleziona Foglio di Lavoro", choices=[], interactive=True)
            fine_tune_button = gr.Button("Avvia Fine-Tuning", interactive=False, visible=False)
            fine_tune_status_output = gr.Textbox(label="Stato del Fine-Tuning", interactive=False, lines=3)
    
    # Event Listeners
    fine_tune_file_input.change(
        fn=lambda f: [gr.Dropdown.update(choices=openpyxl.load_workbook(f.name).sheetnames, interactive=True), gr.Button.update(interactive=True, visible=True), "File caricato. Seleziona un foglio e avvia il fine-tuning."],
        inputs=[fine_tune_file_input],
        outputs=[fine_tune_sheet_dropdown, fine_tune_button, fine_tune_status_output]
    )
    
    fine_tune_button.click(
        fn=fine_tune_model,
        inputs=[fine_tuning_state, fine_tune_file_input, fine_tune_sheet_dropdown],
        outputs=[fine_tune_status_output]
    )
    
    # Nota: Ho rimosso la parte di "generazione su file" e "generazione singola"
    # come da tua richiesta, per concentrarci solo sul fine-tuning.
    # L'implementazione futura per la generazione andrà in un altro file.

if __name__ == "__main__":
    demo.launch()

```python
# excel_reader.py
# Modulo per la gestione avanzata dei file Excel, con implementazione di chunking e pulizia dati.

import pandas as pd
import openpyxl
from transformers import AutoTokenizer

# Configurazione globale per il tokenizer, usata anche qui
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 512

def detect_header_row(df_sheet):
    """
    Individua la riga di intestazione (header) basandosi sull'assenza di valori numerici.
    
    Questa funzione analizza le prime 10 righe di un DataFrame e restituisce l'indice
    della prima riga che non contiene solo valori numerici.
    Questo aiuta a gestire i file Excel che hanno metadati o righe vuote prima dell'intestazione.
    """
    for i in range(min(10, len(df_sheet))):
        if not df_sheet.iloc[i].apply(lambda x: isinstance(x, (int, float))).all():
            return i
    return 0

def find_giudizio_column(df):
    """
    Cerca la colonna 'Giudizio' basandosi su una logica robusta.
    Prima controlla la colonna 'H' (indice 7). Se non la trova, cerca in tutte le colonne.
    """
    # Cerca la colonna 'Giudizio' in modo case-insensitive
    giudizio_col = None
    for col in df.columns:
        if isinstance(col, str) and col.strip().lower() == 'giudizio':
            giudizio_col = col
            break
            
    # Se la colonna non è stata trovata, solleva un errore
    if giudizio_col is None:
        raise ValueError("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che il nome della colonna sia 'Giudizio' (o un'altra variante sensibile al maiuscolo/minuscolo).")
        
    return giudizio_col

def chunk_text_with_overlap(text, tokenizer, max_len=MAX_LEN, overlap=50):
    """
    Divide un testo in 'chunk' di dimensioni massime con una sovrapposizione.
    
    Args:
        text (str): Il testo da dividere.
        tokenizer: Il tokenizer del modello.
        max_len (int): La lunghezza massima di ogni chunk in token.
        overlap (int): Il numero di token di sovrapposizione tra i chunk.
        
    Returns:
        Un dizionario con 'input_ids' e 'attention_mask' per ogni chunk.
    """
    if not text:
        return {'input_ids': [], 'attention_mask': []}

    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    chunks = []
    # Gestisce il caso in cui il testo è già più corto del max_len
    if len(tokens) <= max_len:
        chunks.append(tokens)
    else:
        # Crea i chunk con la sovrapposizione
        step = max_len - overlap
        for i in range(0, len(tokens), step):
            chunk = tokens[i:i + max_len]
            chunks.append(chunk)

    # Tokenizza e formatta per il modello
    model_inputs = tokenizer.pad({'input_ids': chunks}, padding=True, return_tensors='pt')
    return model_inputs

def load_and_prepare_excel(file_path, sheet_name):
    """
    Carica un file Excel e prepara i dati per l'addestramento.
    
    Questa funzione legge il foglio di lavoro selezionato, identifica l'intestazione,
    trova le colonne 'Giudizio' e le altre, e prepara un DataFrame.
    """
    try:
        # Legge il foglio di lavoro specifico con il nome fornito
        df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Troviamo la riga di intestazione
        header_row_index = detect_header_row(df_sheet)
        
        # Assegniamo l'intestazione e rimuoviamo le righe sopra di essa
        df_sheet.columns = df_sheet.iloc[header_row_index]
        df_sheet = df_sheet[header_row_index + 1:].reset_index(drop=True)
        
        # Rimuove le righe vuote
        df_sheet.dropna(how='all', inplace=True)

        # Trova la colonna 'Giudizio'
        giudizio_col = find_giudizio_column(df_sheet)
        
        # Identifica le altre colonne che devono essere usate per il prompt
        other_cols = [col for col in df_sheet.columns if col != giudizio_col]
        
        # Prepara la lista di dizionari per la creazione del dataset
        data_for_dataset = []
        for index, row in df_sheet.iterrows():
            prompt_parts = []
            for col in other_cols:
                value = row.get(col)
                if pd.notna(value) and str(value).strip():
                    prompt_parts.append(f"{col}: {str(value).strip()}")
            
            prompt_text = " ".join(prompt_parts)
            target_text = str(row[giudizio_col]).strip() if pd.notna(row[giudizio_col]) else ""

            if prompt_text and target_text:
                data_for_dataset.append({
                    'input_text': prompt_text,
                    'target_text': target_text
                })
        
        if not data_for_dataset:
            return pd.DataFrame() # Restituisce un DataFrame vuoto se non ci sono dati validi
            
        return pd.DataFrame(data_for_dataset)
        
    except Exception as e:
        print(f"Errore nel caricamento del file Excel: {e}")
        return None
```text
# requirements.txt
pandas
openpyxl
datasets
transformers
peft
torch
gradio
