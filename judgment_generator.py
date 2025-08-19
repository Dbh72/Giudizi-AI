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
        max_length (int): La lunghezza massima di input per ogni chunk.
        chunk_overlap (int): La sovrapposizione tra i chunk.

    Returns:
        str: La risposta generata dal modello.
    """
    # Tokenizza il testo di input in un formato che il modello può usare
    tokens = tokenizer(input_text, return_tensors="pt", truncation=False)["input_ids"]
    
    # Se il testo non è troppo lungo, lo elabora direttamente
    if tokens.shape[1] <= max_length:
        outputs = model.generate(
            tokens.to(model.device),
            max_length=max_length,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Logica per l'elaborazione a chunk
    input_chunks = []
    start = 0
    end = max_length
    while start < tokens.shape[1]:
        chunk = tokens[0, start:end]
        input_chunks.append(chunk)
        start += (max_length - chunk_overlap)
        end = start + max_length
        if end > tokens.shape[1]:
            end = tokens.shape[1]
            if start >= end:
                break
    
    # Logga che stiamo per iniziare l'elaborazione a chunk
    progress_container(f"Input troppo lungo ({tokens.shape[1]} token), verrà spezzato in {len(input_chunks)} chunk.", "info")
    
    # Genera una risposta per ogni chunk
    generated_texts = []
    for i, chunk in enumerate(input_chunks):
        progress_container(f"Generazione per il chunk {i+1} su {len(input_chunks)}...", "info")
        outputs = model.generate(
            chunk.unsqueeze(0).to(model.device),
            max_length=max_length,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        generated_texts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    # Ricombina le risposte dei vari chunk
    return " ".join(generated_texts)


def load_model(model_path, progress_container):
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


def generate_judgments_for_file(df, model, tokenizer, progress_container, sheet_name, giudizio_col_name):
    """
    Genera i giudizi per ogni riga di un DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame con i dati da elaborare.
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        progress_container (callable): Funzione per inviare messaggi di stato.
        sheet_name (str): Il nome del foglio di lavoro corrente.
        giudizio_col_name (str): Il nome della colonna 'Giudizio'.
    
    Returns:
        pd.DataFrame: Il DataFrame aggiornato con i giudizi generati.
    """
    try:
        progress_container(f"Inizio generazione dei giudizi per il foglio '{sheet_name}'.", "info")
        
        # Determina da dove riprendere il processo
        start_index = 0
        if giudizio_col_name in df.columns:
            # Trova l'ultima riga non nulla nella colonna 'Giudizio'
            last_judged_row = df[df[giudizio_col_name].notna()].index.max()
            if not pd.isna(last_judged_row):
                start_index = last_judged_row + 1
                progress_container(f"Ripresa della generazione dall'indice {start_index} per la colonna '{giudizio_col_name}'.", "info")

        if start_index >= len(df):
            progress_container("Tutti i giudizi sembrano già essere stati generati. Processo completato.", "success")
            return df

        # Itera sulle righe del DataFrame
        for i in range(start_index, len(df)):
            row = df.iloc[i]
            
            # Costruisci l'input per il modello
            prompt = " ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notna(row[col]) and col != giudizio_col_name and not col.lower().startswith(('alunno', 'assenti', 'cnt'))])
            
            progress_container(f"Generazione per la riga {i+1}...", "info")
            
            # Controlla la lunghezza dell'input e usa il chunking se necessario
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
