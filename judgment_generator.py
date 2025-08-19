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
    if not input_text.strip():
        return ""
        
    tokens = tokenizer.encode(input_text, return_tensors='pt', max_length=10000, truncation=True)
    num_tokens = tokens.shape[1]
    
    # Se il testo è già entro i limiti, lo processa direttamente
    if num_tokens <= max_length:
        return generate_single_judgment(model, tokenizer, input_text)
    
    all_responses = []
    
    # Suddivisione in chunk
    for i in range(0, num_tokens, max_length - chunk_overlap):
        chunk_tokens = tokens[:, i:i + max_length]
        
        # Decodifica il chunk di token in testo
        chunk_text = tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
        
        # Genera la risposta per il chunk
        response = generate_single_judgment(model, tokenizer, chunk_text)
        all_responses.append(response)
        
    # Combina le risposte dei chunk. Questo è un approccio semplice.
    # Per risposte complesse, potrebbe essere necessaria una logica più avanzata.
    combined_response = " ".join(all_responses)
    return combined_response


def generate_single_judgment(model, tokenizer, input_text):
    """
    Genera un singolo giudizio utilizzando il modello.

    Args:
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        input_text (str): Il testo di input da elaborare.

    Returns:
        str: Il testo del giudizio generato.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(device)
        
        # Generazione della risposta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,  # Aumenta per giudizi più lunghi
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decodifica il testo generato
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output
        
    except Exception as e:
        return f"Errore durante la generazione: {e}"


def load_finetuned_model(model_path, progress_container):
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

def find_last_processed_row(df, column_name="Giudizio"):
    """
    Trova l'indice dell'ultima riga elaborata nella colonna 'Giudizio'
    per riprendere il processo.
    """
    # Normalizza il nome della colonna per la ricerca
    giudizio_col = next((col for col in df.columns if re.search(r'giudizio|valutazione', col, re.IGNORECASE)), None)
    
    if giudizio_col:
        # Trova l'indice dell'ultima riga con un valore non vuoto
        last_filled_index = df[df[giudizio_col].notna() & (df[giudizio_col].astype(str).str.strip() != '')].index.max()
        
        if pd.isna(last_filled_index):
            return 0, df
        else:
            return last_filled_index + 1, df
    
    return 0, df

def generate_judgments(model, tokenizer, df_to_complete, progress_container, start_row=0):
    """
    Genera i giudizi per un DataFrame, riprendendo da un indice specifico.
    """
    giudizio_col = next((col for col in df_to_complete.columns if re.search(r'giudizio|valutazione', col, re.IGNORECASE)), None)
    
    if giudizio_col is None:
        progress_container("Errore: La colonna 'Giudizio' non è stata trovata nel DataFrame.", "error")
        return df_to_complete

    # Esclude le colonne che non servono nel prompt
    exclude_cols = ['alunno', 'assenti', 'cnt']
    input_cols = [col for col in df_to_complete.columns if col.lower() not in exclude_cols and col != giudizio_col]
    
    total_rows = len(df_to_complete)
    
    for i in range(start_row, total_rows):
        row = df_to_complete.iloc[i]
        
        # Controlla se il campo 'Giudizio' è già compilato
        if pd.notna(row[giudizio_col]) and str(row[giudizio_col]).strip() != '':
            progress_container(f"Riga {i+1}/{total_rows}: Giudizio già presente. Saltato.", "info")
            continue

        try:
            input_data = row[input_cols].astype(str, errors='ignore').to_dict()
            prompt = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])
            
            progress_container(f"Riga {i+1}/{total_rows}: Generazione del giudizio...", "info")
            generated_text = generate_single_judgment(model, tokenizer, prompt)
            
            # Assicura che la colonna sia di tipo stringa
            df_to_complete.loc[i, giudizio_col] = str(generated_text)
            
            progress_container(f"Riga {i+1}/{total_rows}: Giudizio generato.", "info")
            time.sleep(1) # per non sovraccaricare il sistema
            
        except Exception as e:
            progress_container(f"Errore nella generazione del giudizio per la riga {i+1}: {e}", "error")
            df_to_complete.loc[i, giudizio_col] = f"ERRORE: {e}"
            continue
            
    return df_to_complete
