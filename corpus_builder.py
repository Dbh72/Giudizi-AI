# ==============================================================================
# File: corpus_builder.py
# Modulo per la gestione del corpus di addestramento.
# Questo file contiene le funzioni per creare e aggiornare il corpus.
# ==============================================================================

import pandas as pd
import os
import shutil

# Definiamo le costanti per il progetto
OUTPUT_DIR = "./modello_finetunato"
CORPUS_FILE = "corpus.csv"

def build_corpus(df, corpus_path):
    """
    Crea un nuovo corpus o aggiorna un corpus esistente con nuovi dati.
    
    Args:
        df (pd.DataFrame): Il DataFrame contenente i nuovi dati.
        corpus_path (str): Il percorso in cui salvare il corpus.
    """
    # Verifichiamo se il file del corpus esiste già
    if os.path.exists(corpus_path):
        # Se esiste, lo carichiamo e ci appendiamo i nuovi dati
        existing_df = pd.read_csv(corpus_path)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        updated_df.to_csv(corpus_path, index=False)
    else:
        # Se non esiste, lo creiamo con i nuovi dati
        df.to_csv(corpus_path, index=False)
    
    print(f"Corpus aggiornato con successo. Dati salvati in: {corpus_path}")

def load_corpus(corpus_path):
    """
    Carica un corpus esistente da un file CSV.
    
    Args:
        corpus_path (str): Il percorso del file CSV del corpus.
        
    Returns:
        pd.DataFrame: Il DataFrame del corpus caricato.
    """
    if os.path.exists(corpus_path):
        return pd.read_csv(corpus_path)
    else:
        return pd.DataFrame() # Restituisce un DataFrame vuoto se il file non esiste
