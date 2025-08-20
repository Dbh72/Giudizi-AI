# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando il modello fine-tunato.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
from transformers import pipeline
import streamlit as st
import warnings

# Ignora i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: FUNZIONI PRINCIPALI
# ==============================================================================

@st.cache_resource
def load_model_for_inference(model, tokenizer):
    """
    Crea una pipeline di generazione di testo.
    
    Args:
        model: Il modello fine-tunato.
        tokenizer: Il tokenizer del modello.
        
    Returns:
        pipeline: La pipeline per la generazione di testo.
    """
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer
    )

def generate_judgments(df, trained_model, tokenizer, progress_container):
    """
    Genera i giudizi per ogni riga di un DataFrame.
    
    Args:
        df (pd.DataFrame): Il DataFrame da processare.
        trained_model: Il modello fine-tunato.
        tokenizer: Il tokenizer del modello.
        progress_container (function): Funzione per mostrare lo stato del processo.
        
    Returns:
        pd.DataFrame: Il DataFrame con la nuova colonna 'Giudizio Generato'.
    """
    
    # Carica il modello in una pipeline per l'inferenza
    generator = load_model_for_inference(trained_model, tokenizer)
    
    # Rimuove le righe con valori mancanti
    df_clean = df.dropna(subset=['Compito Svolto'])
    
    # Prepara il testo di input per il modello
    inputs = [f"Scrivi un giudizio per il seguente compito: {text}" for text in df_clean['Compito Svolto']]
    
    # Genera i giudizi con un indicatore di progresso
    judgments = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_rows = len(inputs)
    
    for i, input_text in enumerate(inputs):
        # Esegui la generazione del testo
        try:
            generated_text = generator(input_text, max_length=128, num_beams=5, early_stopping=True)
            judgment = generated_text[0]['generated_text']
        except Exception as e:
            judgment = f"Errore nella generazione: {e}"
        
        judgments.append(judgment)
        
        # Aggiorna la barra di progresso
        progress = (i + 1) / total_rows
        progress_bar.progress(progress)
        status_text.text(f"Progresso: {i+1}/{total_rows}")
        
    # Aggiunge i giudizi generati al DataFrame
    df_clean.loc[:, 'Giudizio Generato'] = judgments
    
    progress_bar.empty()
    status_text.empty()
    
    progress_container(st.empty(), "Generazione dei giudizi completata.", "success")
    
    return df_clean
