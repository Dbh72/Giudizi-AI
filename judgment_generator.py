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
import requests
import re
from io import BytesIO

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
        progress_container (callable): Funzione per inviare messaggi di stato.
        max_length (int): La lunghezza massima di ogni chunk.
        chunk_overlap (int): L'overlap tra i chunk per mantenere il contesto.
        
    Returns:
        str: Il testo generato riassemblato.
    """
    # Tokenizza l'input
    input_ids = tokenizer.encode(input_text, return_tensors='pt', truncation=False).to(model.device)
    
    # Calcola il numero di chunk
    num_chunks = int(torch.ceil(torch.tensor(input_ids.shape[1] / (max_length - chunk_overlap))))
    
    generated_parts = []
    
    for i in range(num_chunks):
        start_index = i * (max_length - chunk_overlap)
        end_index = min(start_index + max_length, input_ids.shape[1])
        
        chunk = input_ids[:, start_index:end_index]
        
        # Genera il testo per il chunk
        outputs = model.generate(
            chunk,
            max_length=max_length,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_parts.append(generated_text)
        progress_container(f"Generato giudizio per il chunk {i+1}/{num_chunks}.", "info")
        
    # Unisce le parti generate
    full_generated_text = " ".join(generated_parts)
    return full_generated_text

def get_gemini_api_key():
    """
    Recupera la chiave API di Gemini dal file di configurazione o dall'ambiente.
    """
    from config import GEMINI_API_KEY
    return GEMINI_API_KEY

def generate_judgment_with_gemini(prompt, progress_container):
    """
    Genera un giudizio utilizzando l'API di Gemini.
    """
    api_key = get_gemini_api_key()
    if not api_key:
        progress_container("Chiave API di Gemini non trovata. Impossibile generare giudizi.", "error")
        return None
        
    headers = {
        'Content-Type': 'application/json',
    }
    
    json_data = {
        'contents': [
            {
                'role': 'user',
                'parts': [
                    {'text': prompt},
                ],
            },
        ],
    }

    try:
        response = requests.post(
            f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}',
            headers=headers,
            json=json_data,
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get('candidates') and result['candidates'][0].get('content'):
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            progress_container(f"Risposta API non valida: {result}", "error")
            return "Errore di generazione."
            
    except requests.exceptions.RequestException as e:
        progress_container(f"Errore nella chiamata all'API di Gemini: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return "Errore di connessione."

# ==============================================================================
# SEZIONE 3: FUNZIONE PRINCIPALE DI GENERAZIONE
# ==============================================================================

def generate_judgments(df, model, tokenizer, sheet_name, progress_container):
    """
    Itera su un DataFrame e genera giudizi per le righe con 'Giudizio' vuoto.

    Args:
        df (DataFrame): Il DataFrame da processare.
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        sheet_name (str): Il nome del foglio di lavoro.
        progress_container (callable): Funzione per inviare messaggi di stato.
        
    Returns:
        DataFrame: Il DataFrame aggiornato con i giudizi generati.
    """
    progress_container(f"Inizio generazione giudizi per il foglio '{sheet_name}'...", "info")
    
    try:
        # Trova la colonna 'Giudizio'
        giudizio_col_name = 'Giudizio'
        
        # Iterazione sulle righe del DataFrame
        for i in range(len(df)):
            descrizione = df.at[i, 'Descrizione']
            giudizio = df.at[i, giudizio_col_name]
            
            # Controlla se il giudizio è vuoto e la descrizione non lo è
            if pd.isna(giudizio) or not str(giudizio).strip() and pd.notna(descrizione) and str(descrizione).strip():
                progress_container(f"Generazione giudizio per la riga {i+1}...", "info")
                
                # Crea il prompt per il modello
                prompt = f"Descrizione: {descrizione}\nGiudizio:"
                
                # Genera il giudizio
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
                df.at[i, giudizio_col_name] = generated_text
                
                # Aggiungi un piccolo delay per non sovraccaricare il sistema
                time.sleep(0.5)

        progress_container(f"Generazione completata per il foglio '{sheet_name}'.", "success")
        return df

    except Exception as e:
        progress_container(f"Errore nella generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return df
