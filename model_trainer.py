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
        progress_container (callable): Funzione per inviare messaggi di stato.
        max_length (int): Lunghezza massima di ogni chunk.
        chunk_overlap (int): Sovrapposizione tra i chunk.

    Returns:
        str: Il testo generato riassemblato.
    """
    try:
        # Tokenizza l'input
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        total_length = input_ids.shape[1]
        
        # Calcola i chunk
        chunks = []
        for i in range(0, total_length, max_length - chunk_overlap):
            chunk = input_ids[0, i:i + max_length]
            chunks.append(chunk)

        generated_texts = []
        for i, chunk in enumerate(chunks):
            progress_container(f"Generazione del chunk {i+1}/{len(chunks)}...", "info")
            outputs = model.generate(
                chunk.to(model.device),
                max_length=max_length,
                num_beams=4,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)

        # Riassembla le risposte. Per ora, una semplice concatenazione.
        return " ".join(generated_texts)
        
    except Exception as e:
        progress_container(f"Errore nel processare i chunk: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return ""


def generate_judgments(df, model, tokenizer, sheet_name, progress_container):
    """
    Genera i giudizi per ogni riga del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati da processare.
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        sheet_name (str): Il nome del foglio di lavoro in cui si sta lavorando.
        progress_container (callable): Funzione per inviare messaggi di stato.
        
    Returns:
        pd.DataFrame: Il DataFrame aggiornato con i giudizi generati.
    """
    try:
        # Trova le colonne necessarie
        processed_df, giudizio_col_name, materia_col, desc_col, error_msg = df, 'Giudizio', 'Materia', 'Descrizione Giudizio', None
        
        if error_msg:
            progress_container(f"Errore: {error_msg}", "error")
            return df
        
        # Filtra le righe dove il giudizio non è ancora stato generato
        df_to_process = df[df[giudizio_col_name].astype(str).str.strip() == '']
        
        if df_to_process.empty:
            progress_container("Tutti i giudizi sembrano già essere stati generati per questo foglio.", "warning")
            return df

        progress_container(f"Avvio della generazione per {len(df_to_process)} giudizi...", "info")
        
        for i, row in df_to_process.iterrows():
            progress_container(f"Generazione giudizio per la riga {i+1}...", "info")
            
            # Costruisci il prompt per il modello
            prompt = f"Materia: {row[materia_col]} - Descrizione Giudizio: {row[desc_col]}"
            
            # Tokenizza l'input
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
