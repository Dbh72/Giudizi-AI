# ==============================================================================
# File: excel_reader.py
# Modulo per la lettura e l'elaborazione dei file Excel.
# ==============================================================================

import pandas as pd
from io import BytesIO
import streamlit as st

def load_excel_with_sheets(uploaded_file):
    """
    Carica un file Excel e restituisce un dizionario di DataFrame,
    uno per ogni foglio di lavoro.
    
    Args:
        uploaded_file (UploadedFile): Il file caricato tramite Streamlit.
        
    Returns:
        dict: Un dizionario dove le chiavi sono i nomi dei fogli e i valori sono i DataFrame.
    """
    # Usiamo BytesIO per leggere il file in memoria
    file_bytes = BytesIO(uploaded_file.getvalue())
    
    # Pandas può leggere automaticamente tutti i fogli di lavoro
    all_sheets = pd.read_excel(file_bytes, sheet_name=None, engine='openpyxl')
    
    return all_sheets

def read_excel_to_df(uploaded_file, progress_container):
    """
    Legge un file Excel caricato e permette all'utente di selezionare
    un foglio di lavoro.
    
    Args:
        uploaded_file (UploadedFile): Il file caricato tramite Streamlit.
        progress_container (function): Funzione per mostrare lo stato del processo.
        
    Returns:
        tuple: Un tuple contenente il DataFrame del foglio selezionato e il nome del foglio.
    """
    try:
        progress_container(st.empty(), "Lettura dei fogli di lavoro...", "info")
        
        # Carica il file e tutti i suoi fogli
        all_sheets = load_excel_with_sheets(uploaded_file)
        
        progress_container(st.empty(), "Fogli di lavoro trovati. Seleziona un foglio per continuare.", "success")
        
        # L'utente seleziona il foglio
        sheet_name = st.selectbox(
            "Seleziona il foglio di lavoro da processare:",
            list(all_sheets.keys())
        )
        
        df = all_sheets[sheet_name]
        
        # Verifica se il DataFrame è vuoto
        if df.empty:
            raise ValueError("Il foglio di lavoro selezionato è vuoto.")
            
        progress_container(st.empty(), f"Foglio '{sheet_name}' caricato con successo.", "success")
        
        return df, sheet_name
        
    except Exception as e:
        progress_container(st.empty(), f"Errore durante la lettura del file Excel: {e}", "error")
        st.error(f"Errore: {e}")
        st.stop() # Interrompe l'esecuzione in caso di errore
        
def convert_to_corpus_format(df):
    """
    Converte un DataFrame nel formato richiesto per il corpus.
    
    Args:
        df (pd.DataFrame): Il DataFrame originale.
        
    Returns:
        pd.DataFrame: Il DataFrame convertito.
    """
    # Rinomina le colonne
    df_corpus = df.rename(columns={
        "Compito Svolto": "input_text", 
        "Giudizio": "target_text"
    })
    
    # Rimuovi le righe con valori mancanti
    df_corpus = df_corpus.dropna(subset=['input_text', 'target_text'])
    
    # Aggiungi un prefisso a 'input_text' per l'addestramento
    df_corpus['input_text'] = "Scrivi un giudizio per il seguente compito: " + df_corpus['input_text']
    
    return df_corpus
