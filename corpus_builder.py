# ==============================================================================
# File: corpus_builder.py
# Modulo per la creazione e la gestione del corpus di addestramento.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import os
import shutil
import traceback
from config import CORPUS_FILE

# ==============================================================================
# SEZIONE 2: FUNZIONI PER LA GESTIONE DEL CORPUS
# ==============================================================================

def build_or_update_corpus(new_df, progress_container):
    """
    Costruisce o aggiorna il corpus di addestramento con un nuovo DataFrame.
    """
    try:
        corpus_df = pd.DataFrame()
        if os.path.exists(CORPUS_FILE):
            progress_container("Corpus esistente trovato. Aggiornamento in corso...", "info")
            corpus_df = pd.read_parquet(CORPUS_FILE)
            progress_container(f"Corpus caricato. Righe totali prima dell'aggiornamento: {len(corpus_df)}", "info")
        else:
            progress_container("Nessun corpus esistente trovato. Verrà creato uno nuovo.", "info")

        progress_container("Lettura del nuovo file di addestramento...", "info")

        if not new_df.empty:
            progress_container(f"Trovate {len(new_df)} nuove righe da aggiungere.", "info")
            corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
            corpus_df.drop_duplicates(inplace=True)
            progress_container(f"Corpus aggiornato. Righe totali: {len(corpus_df)}", "success")
            corpus_df.to_parquet(CORPUS_FILE, index=False)
            progress_container("Corpus salvato con successo.", "success")
        else:
            progress_container("Nessun dato valido nel nuovo file. Il corpus non è stato aggiornato.", "warning")

        return corpus_df

    except Exception as e:
        progress_container(f"Errore durante l'aggiornamento del corpus: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()

def delete_corpus(progress_container):
    """
    Elimina il file del corpus di addestramento.
    """
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)
        progress_container("Corpus di addestramento eliminato.", "success")
    else:
        progress_container("Nessun corpus da eliminare.", "warning")

def load_corpus(progress_container):
    """
    Carica il corpus esistente.
    """
    try:
        if os.path.exists(CORPUS_FILE):
            progress_container("Caricamento del corpus di addestramento...", "info")
            corpus_df = pd.read_parquet(CORPUS_FILE)
            progress_container(f"Corpus caricato. Righe totali: {len(corpus_df)}", "success")
            return corpus_df
        else:
            progress_container("Nessun corpus di addestramento trovato.", "warning")
            return pd.DataFrame()
    except Exception as e:
        progress_container(f"Errore durante il caricamento del corpus: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()
