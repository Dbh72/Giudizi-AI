# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando un modello fine-tuned.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
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
from config import OUTPUT_DIR, MODEL_NAME

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
        max_length (int): La lunghezza massima di input per ogni chunk.
        chunk_overlap (int): La sovrapposizione tra i chunk.

    Returns:
        str: Il testo della risposta generata.
    """
    # Tokenizza l'input
    tokens = tokenizer.encode(input_text, return_tensors="pt")
    
    # Se il testo è già abbastanza corto, generiamo direttamente
    if tokens.size(1) <= max_length:
        input_ids = tokens.to(model.device)
        output_ids = model.generate(input_ids)
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Altrimenti, suddividiamo in chunk
    chunk_size = max_length - chunk_overlap
    chunks = []
    
    for i in range(0, tokens.size(1), chunk_size):
        chunk = tokens[:, i:i + max_length]
        chunks.append(chunk)
    
    generated_texts = []
    for chunk in chunks:
        input_ids = chunk.to(model.device)
        output_ids = model.generate(input_ids)
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
    # Assembla le risposte (logica semplificata, può essere migliorata)
    return " ".join(generated_texts)

def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, output_dir, progress_container):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro.
        output_dir (str): La directory del modello.
        progress_container (callable): Funzione per inviare messaggi di stato.
        
    Returns:
        pd.DataFrame: Il DataFrame completato.
    """
    state_file = os.path.join(output_dir, f"state_{selected_sheet}.json")
    
    # Carica lo stato precedente, se esiste
    start_index = 0
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                start_index = state.get("last_completed_index", 0) + 1
                progress_container(f"Trovato stato precedente. Riprendo la generazione dalla riga {start_index + 2}.", "info")
        except (IOError, json.JSONDecodeError) as e:
            progress_container(f"Errore nel caricamento del file di stato: {e}. Riavvio la generazione dall'inizio.", "warning")
            start_index = 0
    
    # Aggiungi una colonna temporanea per lo stato di generazione
    if 'generation_status' not in df_to_complete.columns:
        df_to_complete['generation_status'] = None
    
    # Aggiungi una colonna temporanea per lo stato di generazione
    df_to_complete['generation_status'] = df_to_complete.apply(
        lambda row: 'completed' if pd.notna(row[giudizio_col]) else None, axis=1
    )
    
    for index, row in df_to_complete.iterrows():
        # Salta le righe già elaborate
        if index < start_index:
            continue
            
        # Salta le righe che hanno già un giudizio
        if row['generation_status'] == 'completed':
            continue

        try:
            # Crea il prompt combinando tutte le colonne tranne 'Giudizio'
            input_data = row.drop(labels=[giudizio_col])
            prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val)])
            
            progress_container(f"Generazione giudizio per riga {index + 2}...", "info")
            
            # Genera il giudizio usando il modello
            inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True).to(model.device)
            
            generated_ids = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )
            
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Assegna il giudizio generato al DataFrame
            df_to_complete.at[index, giudizio_col] = generated_text
            
            # Aggiorna lo stato per il checkpoint
            df_to_complete.at[index, 'generation_status'] = 'completed'
            
            # Salva lo stato
            state = {"last_completed_index": index}
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
        except Exception as e:
            error_message = f"Errore durante la generazione per la riga {index + 2}: {e}"
            progress_container(error_message, "error")
            progress_container(f"Traceback: {traceback.format_exc()}", "error")
            continue
    
    # Rimuove la colonna di stato e il file di stato una volta completata la generazione
    progress_container("Generazione completata con successo!", "success")
    if os.path.exists(state_file):
        os.remove(state_file)
    return df_to_complete.drop(columns=['generation_status'])

def load_trained_model(model_path, progress_container):
    """
    Carica il modello e il tokenizer fine-tuned, controllando se il percorso
    esiste localmente.

    Args:
        model_path (str): Il percorso della directory del modello salvato.
        progress_container (callable): Funzione per inviare messaggi di stato.

    Returns:
        tuple: (model, tokenizer) o (None, None) se il caricamento fallisce.
    """
    try:
        progress_container(f"Caricamento del modello da: {model_path}...", "info")
        
        # Controlla se il modello esiste localmente
        if not os.path.exists(model_path):
            progress_container(f"Errore: La directory del modello '{model_path}' non è stata trovata localmente.", "error")
            return None, None
            
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carica il modello base e applica gli adattatori PEFT
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        progress_container(f"Modello e tokenizer caricati con successo da: {model_path}", "success")
        return model, tokenizer

    except Exception as e:
        progress_container(f"Errore nel caricamento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

