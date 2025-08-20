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

def _process_text_in_chunks(model, tokenizer, input_text, progress_container, max_length=512, chunk_overlap=50):
    """
    Processa un testo di input troppo lungo suddividendolo in chunk e generando
    una risposta per ogni chunk, poi riassembla le risposte.

    Args:
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        input_text (str): Il testo di input da elaborare.
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
        max_length (int): La lunghezza massima di un chunk.
        chunk_overlap (int): Il numero di token di sovrapposizione tra i chunk.

    Returns:
        str: Il testo generato riassemblato.
    """
    # Tokenizza il testo di input
    input_tokens = tokenizer(input_text, return_tensors="pt", truncation=False, max_length=None)
    
    # Se il testo è troppo lungo, lo suddividi in chunk
    chunk_size = max_length - chunk_overlap
    input_ids = input_tokens.input_ids[0]
    
    all_generated_texts = []
    
    for i in range(0, len(input_ids), chunk_size):
        chunk = input_ids[i:i + max_length]
        
        # Genera il testo per il chunk
        outputs = model.generate(
            chunk.to(model.device).unsqueeze(0),
            max_length=max_length,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_generated_texts.append(generated_text)
    
    # Riassembla i testi generati
    return " ".join(all_generated_texts)

# ==============================================================================
# SEZIONE 3: FUNZIONI PRINCIPALI
# ==============================================================================

def generate_judgments_and_save(df, model, tokenizer, sheet_name, progress_container):
    """
    Genera giudizi per le righe mancanti in un DataFrame e restituisce il DataFrame aggiornato.

    Args:
        df (pd.DataFrame): Il DataFrame con le righe da processare.
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        sheet_name (str): Il nome del foglio di lavoro.
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.

    Returns:
        pd.DataFrame: Il DataFrame aggiornato con i giudizi generati.
    """
    try:
        progress_container(f"Inizio generazione dei giudizi per il foglio '{sheet_name}'...", "info")
        
        # Recupera la colonna dei giudizi
        giudizio_col_name = None
        for col in df.columns:
            if re.search(r'giudizio', col, re.IGNORECASE):
                giudizio_col_name = col
                break
        
        if giudizio_col_name is None:
            progress_container(f"Errore: Colonna 'Giudizio' non trovata nel DataFrame.", "error")
            return df
        
        # Crea una copia per evitare modifiche dirette al DataFrame originale
        df_copy = df.copy()

        # Itera su ogni riga del DataFrame
        for i, row in df_copy.iterrows():
            # Controlla se la cella del giudizio è vuota
            if pd.notna(row[giudizio_col_name]):
                continue

            # Costruisci il prompt per il modello
            input_parts = []
            exclude_cols_regex = r'pos|posizione|alunno|assenti|cnt|giudizio'
            for col in df_copy.columns:
                if not re.search(exclude_cols_regex, col, re.IGNORECASE):
                    value = row[col]
                    if pd.notna(value) and str(value).strip() != '':
                        input_parts.append(f"{col}: {value}")
            
            prompt = " ".join(input_parts)
            
            if not prompt:
                progress_container(f"Riga {i+1} saltata: prompt vuoto.", "warning")
                continue

            # Genera il giudizio usando il modello
            input_tokens = tokenizer(prompt, return_tensors="pt")
            max_model_length = 512
            
            if input_tokens.input_ids.shape[1] > max_model_length:
                # Usa la funzione di chunking per non perdere dati, passando il progress_container
                generated_text = _process_text_in_chunks(model, tokenizer, prompt, progress_container)
            else:
                # Usa la generazione standard se l'input non è troppo lungo
                outputs = model.generate(
                    input_tokens.input_ids.to(model.device),
                    max_length=max_model_length,
                    num_beams=4,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    early_stopping=True
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Aggiungi il giudizio generato al DataFrame
            df_copy.at[i, giudizio_col_name] = generated_text
            
            # Aggiorna l'interfaccia utente
            progress_container(f"Giudizio generato per la riga {i+1}.", "info")
            
            # Aggiungi un piccolo delay per non sovraccaricare il sistema
            time.sleep(0.5)

        progress_container(f"Generazione completata per il foglio '{sheet_name}'.", "success")
        return df_copy

    except Exception as e:
        progress_container(f"Errore nella generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()
