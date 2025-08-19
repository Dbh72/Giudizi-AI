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
    """
    if len(tokenizer.tokenize(input_text)) <= max_length:
        return _generate_judgment(model, tokenizer, input_text)
    
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    chunks = []
    
    for i in range(0, len(tokens), max_length - chunk_overlap):
        chunk_tokens = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk_tokens))

    generated_judgments = [_generate_judgment(model, tokenizer, chunk) for chunk in chunks]
    return " ".join(generated_judgments)


def _generate_judgment(model, tokenizer, input_text):
    """
    Genera un singolo giudizio basato su un testo di input.
    """
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def load_model(model_path, progress_container):
    """
    Carica il modello e il tokenizer fine-tuned, controllando se il percorso
    esiste localmente.
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
        progress_container(f"Errore durante il caricamento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None


def generate_judgments_for_excel(model, tokenizer, file_object, sheet_name, progress_container):
    """
    Genera giudizi per un file Excel utilizzando un modello fine-tuned e salva
    il risultato in una nuova colonna.
    """
    try:
        progress_container("Lettura del file Excel per la generazione dei giudizi...", "info")
        
        # Usa file_object direttamente per leggere il dataframe
        df = pd.read_excel(file_object, sheet_name=sheet_name, header=None)
        
        # Trova la riga di intestazione e le colonne
        original_df, giudizio_col_name = find_header_row_and_columns(df)
        
        # Se la colonna "Giudizio" non è stata trovata, la crea
        if giudizio_col_name is None:
            progress_container("Colonna 'Giudizio' non trovata. Creazione di una nuova colonna.", "warning")
            giudizio_col_name = 'Giudizio'
            original_df[giudizio_col_name] = ""
        else:
            progress_container("Colonna 'Giudizio' trovata. La userò per la scrittura.", "info")

        # Inizializza la colonna dei giudizi se non esiste o è vuota
        if giudizio_col_name not in original_df.columns:
            original_df[giudizio_col_name] = ""
        
        # Cicla sulle righe del DataFrame
        for index, row in original_df.iterrows():
            if pd.notna(row[giudizio_col_name]) and str(row[giudizio_col_name]).strip() != '':
                progress_container(f"Riga {index+2}: Giudizio già esistente. Saltato.", "info")
                continue

            input_data = row.drop(labels=[c for c in original_df.columns if isinstance(c, str) and ('giudizio' in c.lower() or 'alunno' in c.lower() or 'assenti' in c.lower() or 'cnt' in c.lower() or 'pos' in c.lower())], errors='ignore')
            prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])

            if not prompt_text.strip():
                progress_container(f"Riga {index+2}: Dati di input insufficienti. Giudizio non generato.", "warning")
                original_df.at[index, giudizio_col_name] = "Giudizio non generato (dati insufficienti)"
                continue

            progress_container(f"Riga {index+2}: Generazione giudizio...", "info")
            generated_judgment = _process_text_in_chunks(model, tokenizer, prompt_text)
            
            # Aggiorna la riga con il giudizio generato
            original_df.at[index, giudizio_col_name] = generated_judgment.strip()
            progress_container(f"Giudizio generato per la riga {index+2}.", "info")

        return original_df
    except Exception as e:
        progress_container(f"Errore critico durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()

def find_header_row_and_columns(df):
    """
    Trova la riga di intestazione e le posizioni della colonna 'Giudizio'.
    """
    try:
        for i in range(min(50, len(df))):
            row = df.iloc[i].astype(str).str.lower()
            if 'giudizio' in row.values:
                header_row = df.iloc[i]
                df.columns = make_columns_unique(header_row.values)
                df = df.iloc[i+1:].reset_index(drop=True)
                giudizio_col = next((col for col in df.columns if isinstance(col, str) and 'giudizio' in col.lower()), None)
                return df, giudizio_col
        return df, None
    except Exception as e:
        print(f"Errore nella ricerca dell'header: {e}")
        return df, None

def make_columns_unique(columns):
    """
    Garantisce che i nomi delle colonne siano unici, aggiungendo un contatore
    se necessario.
    """
    seen = {}
    new_columns = []
    for col in columns:
        original_col = col
        if original_col in seen:
            seen[original_col] += 1
            new_columns.append(f"{original_col}_{seen[original_col]}")
        else:
            seen[original_col] = 0
            new_columns.append(original_col)
    return new_columns
