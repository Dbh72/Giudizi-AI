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
        max_length (int): Lunghezza massima del chunk.
        chunk_overlap (int): Lunghezza della sovrapposizione tra i chunk.
        
    Returns:
        str: Il testo generato riassemblato.
    """
    tokens = tokenizer(input_text, return_tensors="pt", truncation=False)
    input_ids = tokens.input_ids[0]
    
    # Calcola i chunk con sovrapposizione
    chunks = []
    for i in range(0, len(input_ids), max_length - chunk_overlap):
        end = min(i + max_length, len(input_ids))
        chunks.append(input_ids[i:end])

    generated_texts = []
    
    for i, chunk in enumerate(chunks):
        progress_container(f"Generazione per il chunk {i+1}/{len(chunks)}...", "info")
        outputs = model.generate(
            chunk.unsqueeze(0).to(model.device),
            max_length=max_length,
            num_beams=4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            early_stopping=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
        
    # Riassembla le risposte. Qui si può implementare una logica più sofisticata.
    # Per ora, le uniamo semplicemente.
    return " ".join(generated_texts)


def load_fine_tuned_model(progress_container):
    """
    Carica un modello fine-tuned e il suo tokenizer da una directory.
    """
    try:
        model_path = os.path.join(OUTPUT_DIR, "final_model")
        progress_container(f"Caricamento del modello fine-tuned da: {model_path}...", "info")
        
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carica il modello base
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        
        # Carica l'adattatore PEFT
        model = PeftModel.from_pretrained(base_model, model_path)
        
        progress_container("Modello e tokenizer caricati con successo.", "success")
        return model, tokenizer
        
    except Exception as e:
        progress_container(f"Errore nel caricamento del modello: {e}. Il modello potrebbe non essere stato ancora addestrato.", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

def generate_judgments_for_sheet(df, sheet_name, progress_container, model, tokenizer):
    """
    Genera i giudizi per un singolo foglio di calcolo, elaborando riga per riga.
    """
    try:
        giudizio_col_name = None
        for col in df.columns:
            cleaned_col = re.sub(r'\s+', '', str(col)).lower()
            if cleaned_col == "giudizio":
                giudizio_col_name = col
                break
        if giudizio_col_name is None:
            progress_container(f"Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'.", "error")
            return df
        
        # Identificazione delle colonne da escludere per il prompt
        excluded_cols = ['alunno', 'assenti', 'cnt', 'pos']
        
        for i, row in df.iterrows():
            if not pd.isna(row[giudizio_col_name]) and str(row[giudizio_col_name]).strip():
                progress_container(f"Salto la riga {i+1}: il giudizio esiste già.", "info")
                continue
            
            # Costruisci il prompt escludendo le colonne non pertinenti
            input_parts = []
            for col in df.columns:
                cleaned_col = re.sub(r'\s+', '', str(col)).lower()
                if cleaned_col not in excluded_cols and col != giudizio_col_name:
                    value = str(row[col]).strip()
                    if value:
                        input_parts.append(f"{col}: {value}")

            prompt = " ".join(input_parts)
            
            if not prompt:
                progress_container(f"Salto la riga {i+1}: prompt vuoto.", "warning")
                continue
            
            progress_container(f"Generazione per la riga {i+1}...", "info")
            
            input_tokens = tokenizer(prompt, return_tensors="pt")
            max_model_length = 512
            
            if input_tokens.input_ids.shape[1] > max_model_length:
                # Usa la funzione di chunking per non perdere dati
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
