# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando il modello fine-tunato.
# ==============================================================================

import streamlit as st
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Definiamo le costanti per il progetto
OUTPUT_DIR = "./modello_finetunato"
MODEL_NAME = "t5-small"

def progress_container_stub(*args, **kwargs):
    """Funzione placeholder per evitare errori se non in ambiente Streamlit."""
    pass

def generate_judgments(df, model, tokenizer, status_placeholder):
    """
    Genera i giudizi per un DataFrame dato utilizzando il modello fine-tunato.
    
    Args:
        df (pd.DataFrame): Il DataFrame da processare.
        model: Il modello fine-tunato.
        tokenizer: Il tokenizer del modello.
        status_placeholder: Il placeholder di Streamlit per mostrare i progressi.
        
    Returns:
        pd.DataFrame: Il DataFrame con la colonna 'Giudizio' aggiunta.
    """
    progress_container = st.session_state.get('progress_container', progress_container_stub)
    
    # Assicurati che le colonne 'source' e 'target' esistano
    if 'source' not in df.columns or 'target' not in df.columns:
        progress_container(status_placeholder, "Errore: Il file Excel deve contenere le colonne 'source' e 'target'.", "error")
        return None

    progress_container(status_placeholder, "Inizio della generazione dei giudizi...", "info")
    
    generated_judgments = []
    total_rows = len(df)
    
    # Usa un iteratore per il progresso
    progress_bar = st.progress(0)
    
    for i, row in df.iterrows():
        source_text = row['source']
        input_ids = tokenizer.encode(source_text, return_tensors='pt')
        
        # Genera il testo
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_judgments.append(generated_text)
        
        # Aggiorna la barra di progresso
        progress_bar.progress((i + 1) / total_rows)
        progress_container(status_placeholder, f"Generazione giudizio {i+1} di {total_rows} completata.", "info")

    df['Giudizio'] = generated_judgments
    progress_container(status_placeholder, "Generazione dei giudizi completata con successo!", "success")
    
    return df
