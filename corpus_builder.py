# ==============================================================================
# File: corpus_builder.py
# Modulo per la creazione e la gestione del corpus di addestramento.
# Si occupa di unire i dati da diverse fonti e di salvarli in un unico dataset.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import os
import shutil
import traceback

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE
# ==============================================================================
# Il nome del file del corpus di addestramento.
CORPUS_FILE = "training_corpus.parquet"

# ==============================================================================
# SEZIONE 3: FUNZIONI PER LA GESTIONE DEL CORPUS
# ==============================================================================

def build_or_update_corpus(new_df, progress_container):
    """
    Costruisce o aggiorna il corpus di addestramento con un nuovo DataFrame.
    Se un corpus esistente è presente, concatena i nuovi dati e rimuove i duplicati.

    Args:
        new_df (pd.DataFrame): Il DataFrame con i nuovi dati da aggiungere.
        progress_container (list): Una lista per i messaggi di progresso.

    Returns:
        pd.DataFrame: Il DataFrame del corpus aggiornato.
    """
    try:
        # Verifica se un corpus esistente è già presente
        if os.path.exists(CORPUS_FILE):
            progress_container.append("Trovato un corpus esistente. Caricamento in corso...")
            corpus_df = pd.read_parquet(CORPUS_FILE)
            progress_container.append(f"Corpus esistente caricato. Righe attuali: {len(corpus_df)}")
        else:
            progress_container.append("Nessun corpus esistente trovato. Verrà creato uno nuovo.")
            corpus_df = pd.DataFrame()

        # Concatena il nuovo DataFrame con il corpus esistente
        if not new_df.empty:
            progress_container.append(f"Trovate {len(new_df)} nuove righe da aggiungere.")
            corpus_df = pd.concat([corpus_df, new_df], ignore_index=True)
            corpus_df.drop_duplicates(inplace=True)
            progress_container.append(f"Corpus aggiornato. Totale righe: {len(corpus_df)}")
            corpus_df.to_parquet(CORPUS_FILE, index=False)
            progress_container.append("Corpus salvato con successo.")
        else:
            progress_container.append("Nessun dato valido nel nuovo file. Il corpus non è stato aggiornato.")

        return corpus_df

    except Exception as e:
        progress_container.append(f"Errore durante l'aggiornamento del corpus: {e}")
        progress_container.append(traceback.format_exc())
        return pd.DataFrame()

def delete_corpus(progress_container):
    """
    Elimina il file del corpus di addestramento.

    Args:
        progress_container (list): Una lista per i messaggi di progresso.
    """
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)
        progress_container.append("Corpus di addestramento eliminato.")
    else:
        progress_container.append("Nessun corpus da eliminare.")

def load_corpus(progress_container):
    """
    Carica il corpus esistente.

    Args:
        progress_container (list): Una lista per i messaggi di progresso.

    Returns:
        pd.DataFrame: Il DataFrame del corpus caricato, o un DataFrame vuoto se non esiste.
    """
    try:
        if os.path.exists(CORPUS_FILE):
            progress_container.append("Caricamento del corpus di addestramento...")
            corpus_df = pd.read_parquet(CORPUS_FILE)
            progress_container.append(f"Corpus caricato. Righe totali: {len(corpus_df)}")
            return corpus_df
        else:
            progress_container.append("Nessun corpus di addestramento trovato.")
            return pd.DataFrame()
    except Exception as e:
        progress_container.append(f"Errore durante il caricamento del corpus: {e}")
        progress_container.append(traceback.format_exc())
        return pd.DataFrame()

