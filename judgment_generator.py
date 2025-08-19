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
    tokens = tokenizer.tokenize(input_text)
    token_chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length - chunk_overlap)]
    
    generated_responses = []
    
    for chunk in token_chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        inputs = tokenizer(chunk_text, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            early_stopping=True
        )
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_responses.append(generated_response)
        
    return " ".join(generated_responses)

def generate_judgments_on_excel(model, tokenizer, uploaded_file, selected_sheet, progress_container):
    """
    Genera i giudizi per un file Excel con una colonna 'Giudizio' vuota.
    """
    try:
        progress_container("Caricamento del file per la generazione...", "info")
        file_bytes = uploaded_file.getvalue()
        df_original = pd.read_excel(BytesIO(file_bytes), sheet_name=selected_sheet)

        # Trova la colonna 'Giudizio' (case-insensitive)
        giudizio_col = next((col for col in df_original.columns if 'giudizio' in str(col).lower()), None)
        if not giudizio_col:
            progress_container("Colonna 'Giudizio' non trovata nel file Excel.", "error")
            return df_original, "Colonna 'Giudizio' non trovata."

        # Identifica le righe con il giudizio vuoto o mancante
        rows_to_process = df_original[df_original[giudizio_col].isnull() | (df_original[giudizio_col] == '')].copy()
        total_rows = len(rows_to_process)

        if total_rows == 0:
            progress_container("Tutti i giudizi sono già compilati. Nessuna riga da processare.", "warning")
            return df_original, "Nessun giudizio da generare."

        progress_container(f"Trovate {total_rows} righe con giudizio vuoto. Avvio generazione...", "info")

        # Esclude le colonne che non servono nel prompt
        exclude_cols = ['alunno', 'assenti', 'cnt', 'pos']
        input_cols = [col for col in rows_to_process.columns if str(col).lower() not in exclude_cols and col != giudizio_col]
        
        # Genera i giudizi riga per riga
        for index, row in rows_to_process.iterrows():
            input_text = " ".join([f"{col}: {str(row[col])}" for col in input_cols if pd.notna(row[col]) and str(row[col]).strip() != ''])
            
            if not input_text.strip():
                progress_container(f"Riga {index + 2}: Salto, dati insufficienti per la generazione.", "warning")
                continue

            # Genera il giudizio usando il modello
            generated_text = _process_text_in_chunks(model, tokenizer, input_text)
            
            # Assegna il giudizio generato al DataFrame
            df_original.at[index, giudizio_col] = generated_text

            progress_container(f"Riga {index + 2}: Giudizio generato.", "info")

        progress_container("Generazione completata con successo.", "success")
        return df_original, "Generazione completata."

    except Exception as e:
        progress_container(f"Errore durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame(), f"Errore: {e}"

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
        progress_container(f"Errore nel caricamento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None
