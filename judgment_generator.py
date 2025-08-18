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
        max_length (int): La lunghezza massima di input che il modello può gestire.
        chunk_overlap (int): Il numero di token che si sovrappongono tra i chunk.

    Returns:
        str: Il giudizio riassemblato.
    """
    tokens = tokenizer.encode(input_text)
    token_chunks = []
    
    # Crea i chunk con sovrapposizione
    for i in range(0, len(tokens), max_length - chunk_overlap):
        chunk = tokens[i:i + max_length]
        token_chunks.append(chunk)

    generated_texts = []
    for chunk in token_chunks:
        # Decodifica il chunk di token in testo
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        # Genera il giudizio per il singolo chunk
        input_ids = tokenizer.encode(chunk_text, return_tensors="pt").to(model.device)
        output_ids = model.generate(input_ids)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
    # Riassembla i giudizi
    # Qui usiamo semplicemente uno spazio per unire le frasi,
    # ma una logica più sofisticata potrebbe essere necessaria a seconda del caso.
    final_judgment = " ".join(generated_texts)
    return final_judgment.strip()


def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, output_dir, progress_container):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro.
        output_dir (str): La directory del modello fine-tuned.
        progress_container (list): Una lista per i messaggi di progresso.

    Returns:
        pd.DataFrame: Il DataFrame completato con i nuovi giudizi.
    """
    state_file = os.path.join(output_dir, f"{selected_sheet}_generation_state.json")
    
    # Aggiungi una colonna di stato per tracciare il progresso, se non esiste
    if 'generation_status' not in df_to_complete.columns:
        df_to_complete['generation_status'] = 'pending'

    # Carica lo stato precedente se esiste
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
        df_to_complete.loc[df_to_complete.index.isin(state['completed_indices']), 'generation_status'] = 'completed'
        progress_container.append(f"Ripresa della generazione. {len(state['completed_indices'])} righe già completate.")
    else:
        # Inizializza il file di stato
        state = {'completed_indices': []}
        with open(state_file, 'w') as f:
            json.dump(state, f)

    # Imposta il modello sulla GPU se disponibile
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Itera solo sulle righe 'pending'
    pending_rows = df_to_complete[df_to_complete['generation_status'] == 'pending'].index
    total_to_process = len(pending_rows)
    processed_count = 0

    progress_container.append(f"Inizio della generazione dei giudizi per {total_to_process} righe...")

    for index in pending_rows:
        try:
            row = df_to_complete.loc[index]
            # Assumiamo che la prima colonna contenente testo sia quella di input
            input_col = df_to_complete.columns[0]
            input_text = str(row[input_col]) if pd.notna(row[input_col]) else ""

            if not input_text:
                continue

            # Check della lunghezza del testo
            tokens = tokenizer.encode(input_text, return_tensors='pt')
            if tokens.size(1) > 512:
                # Se il testo è troppo lungo, lo processa in chunk
                generated_judgment = _process_text_in_chunks(model, tokenizer, input_text)
            else:
                # Generazione standard
                input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
                output_ids = model.generate(input_ids)
                generated_judgment = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Aggiunge il giudizio al DataFrame
            df_to_complete.loc[index, giudizio_col] = generated_judgment
            df_to_complete.loc[index, 'generation_status'] = 'completed'

            # Aggiorna lo stato e il file di stato
            state['completed_indices'].append(index)
            with open(state_file, 'w') as f:
                json.dump(state, f)

            processed_count += 1
            progress_container.append(f"Generato giudizio per la riga {index + 2}. Progresso: {processed_count}/{total_to_process}")

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
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"Modello e tokenizer caricati con successo.")
        return model, tokenizer
        
    except Exception as e:
        print(f"Errore nel caricare il modello o il tokenizer: {e}")
        traceback.print_exc()
        return None, None

