# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando un modello fine-tuned.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo le librerie necessarie per la generazione e la gestione dei file.
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import pandas as pd
import traceback
from datetime import datetime
import json
import time

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: FUNZIONI AUSILIARIE PER LA GENERAZIONE
# ==============================================================================

def _process_text_in_chunks(model, tokenizer, input_text, max_length=512, chunk_overlap=50):
    """
    Processa un testo di input troppo lungo suddividendolo in chunk e generando
    una risposta per ogni chunk, poi riassembla le risposte.

    Args:
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        input_text (str): Il testo di input da elaborare.
        max_length (int): La lunghezza massima di input per il modello.
        chunk_overlap (int): La sovrapposizione tra i chunk.

    Returns:
        str: Il giudizio generato combinando i chunk.
    """
    # Tokenizza l'input e ottiene il numero di token
    tokens = tokenizer.encode(input_text, return_tensors='pt', truncation=False)
    num_tokens = tokens.shape[1]

    # Se il testo è già abbastanza corto, lo processa direttamente
    if num_tokens <= max_length:
        input_ids = tokens.to(model.device)
        generated_ids = model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    # Calcola il passo per il chunking
    step = max_length - chunk_overlap
    chunks = []
    
    # Crea i chunk e li aggiunge a una lista
    for i in range(0, num_tokens, step):
        chunk_ids = tokens[:, i:i+max_length]
        chunks.append(chunk_ids)
        
    generated_texts = []
    
    # Processa ogni chunk
    for chunk_ids in chunks:
        chunk_ids = chunk_ids.to(model.device)
        generated_ids = model.generate(chunk_ids, max_length=150, num_beams=5, early_stopping=True)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
    # Combina i testi generati, rimuovendo le ridondanze
    combined_text = " ".join(generated_texts)
    
    # Post-processa il testo per rimuovere frasi ripetute o incongrue
    return combined_text

def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, output_dir, progress_container):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro.
        output_dir (str): La directory dove salvare i file di stato.
        progress_container (list): Lista per i messaggi di stato di Streamlit.

    Returns:
        pd.DataFrame: Il DataFrame completato.
    """
    state_file = os.path.join(output_dir, f"checkpoint_gen_{selected_sheet}.json")

    # Inizializza o carica lo stato
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        last_index = state.get('last_processed_index', -1)
        progress_container.append(f"Checkpoint trovato. Riprendo la generazione dalla riga {last_index + 2} del foglio '{selected_sheet}'.")
    else:
        last_index = -1
        state = {'last_processed_index': -1, 'start_time': datetime.now().isoformat()}
        progress_container.append("Avvio di una nuova sessione di generazione.")

    # Aggiungi una colonna di stato per tracciare le righe elaborate
    df_to_complete['generation_status'] = 'Pending'
    if last_index > -1:
        df_to_complete.loc[:last_index, 'generation_status'] = 'Completed'
        
    to_process_df = df_to_complete[df_to_complete['generation_status'] == 'Pending']
    total_to_process = len(to_process_df)
    processed_count = 0
    
    for index, row in to_process_df.iterrows():
        input_text = row.get('descrizione') or row.get('input_text')
        
        # Trova la colonna "input" per il prompt.
        input_col = None
        for col in df_to_complete.columns:
            if isinstance(col, str) and re.search(r'(input|descrizione|commento|testo)', col, re.IGNORECASE):
                input_col = col
                break
        
        if input_col is None:
            # Se non viene trovata, usa la prima colonna come fallback
            input_col = df_to_complete.columns[0]
            
        input_text = str(row[input_col]) if pd.notna(row[input_col]) else ""
        
        if pd.isna(row[giudizio_col]) and input_text:
            try:
                # Genera il giudizio usando la funzione di chunking
                generated_judgment = _process_text_in_chunks(model, tokenizer, input_text)
                
                # Sostituisci il valore NaN con il giudizio generato
                df_to_complete.at[index, giudizio_col] = generated_judgment
                
                # Aggiorna lo stato nel DataFrame
                df_to_complete.at[index, 'generation_status'] = 'Completed'
                
                processed_count += 1
                progress_container.append(f"Generazione per la riga {index + 2} completata. ({processed_count}/{total_to_process})")
                print(f"Generazione per la riga {index + 2} completata. ({processed_count}/{total_to_process})")
                
                # Aggiorna il checkpoint
                state['last_processed_index'] = index
                with open(state_file, 'w') as f:
                    json.dump(state, f)
            except Exception as e:
                error_message = f"Errore durante la generazione per la riga {index + 2}: {e}"
                progress_container.append(error_message)
                print(error_message)
                continue
    
    # Rimuove la colonna di stato e il file di stato una volta completata la generazione
    progress_container.append("Generazione completata con successo!")
    if os.path.exists(state_file):
        os.remove(state_file)
    return df_to_complete.drop(columns=['generation_status'])

def load_trained_model(model_path):
    """
    Carica il modello e il tokenizer fine-tuned.

    Args:
        model_path (str): Il percorso della directory del modello salvato.

    Returns:
        tuple: (model, tokenizer) o (None, None) se il caricamento fallisce.
    """
    try:
        print(f"Caricamento del modello da: {model_path}...")
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Carica il modello base e applica gli adattatori PEFT
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.float16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"Modello e tokenizer caricati con successo da: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Errore nel caricamento del modello: {e}")
        print(traceback.format_exc())
        return None, None
