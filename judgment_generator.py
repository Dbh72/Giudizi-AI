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
        progress_container (callable): Funzione per inviare messaggi di progresso.
        max_length (int): La lunghezza massima di un chunk.
        chunk_overlap (int): Il numero di token che si sovrappongono tra i chunk.

    Returns:
        str: Il testo generato riassemblato.
    """
    try:
        input_tokens = tokenizer(input_text, return_tensors="pt", truncation=False)
        input_ids = input_tokens.input_ids[0]
        
        chunks = []
        start = 0
        while start < len(input_ids):
            end = min(start + max_length, len(input_ids))
            chunks.append(input_ids[start:end])
            if end == len(input_ids):
                break
            start += max_length - chunk_overlap
            
        generated_texts = []
        for i, chunk in enumerate(chunks):
            progress_container(f"Generazione per il chunk {i+1}/{len(chunks)}...", "info")
            chunk_input_ids = chunk.unsqueeze(0).to(model.device)
            outputs = model.generate(
                chunk_input_ids,
                max_length=max_length,
                num_beams=4,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2,
                early_stopping=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
        return " ".join(generated_texts)
    
    except Exception as e:
        progress_container(f"Errore nel processare il testo in chunk: {e}", "error")
        return ""

# ==============================================================================
# SEZIONE 3: FUNZIONE PRINCIPALE PER LA GENERAZIONE
# ==============================================================================

def generate_judgments(df, model, tokenizer, giudizio_col_name, input_cols_name, sheet_name, progress_container):
    """
    Genera i giudizi per le righe di un DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame con i dati da processare.
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        giudizio_col_name (str): Il nome della colonna in cui inserire i giudizi.
        input_cols_name (list): Lista dei nomi delle colonne di input.
        sheet_name (str): Il nome del foglio di lavoro.
        progress_container (callable): Funzione per inviare messaggi di progresso.

    Returns:
        pd.DataFrame: Il DataFrame aggiornato con i nuovi giudizi.
    """
    try:
        progress_container(f"Inizio generazione giudizi per il foglio '{sheet_name}'...", "info")
        df[giudizio_col_name] = ""
        
        for i, row in df.iterrows():
            progress_container(f"Generazione giudizio per la riga {i+1}/{len(df)}...", "info")
            
            # Costruisci il prompt combinando le colonne di input
            prompt = f"generare un giudizio per il seguente testo: "
            for col in input_cols_name:
                prompt += f" {row[col]}"
            
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
        return pd.DataFrame()
