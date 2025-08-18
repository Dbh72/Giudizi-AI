# ==============================================================================
# File: excel_reader.py
# Logica per la preparazione dei dati da file Excel
# ==============================================================================
import pandas as pd
import openpyxl
import re
from io import BytesIO
import traceback
import os
import shutil

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def find_giudizio_column(df):
    """
    Trova la colonna 'Giudizio' nel DataFrame, cercando in modo case-insensitive
    in tutte le intestazioni.

    Args:
        df (pd.DataFrame): Il DataFrame del foglio da analizzare.

    Returns:
        str: Il nome della colonna 'Giudizio' o None se non trovata.
    """
    # Cerca la parola 'giudizio' in modo case-insensitive tra le colonne.
    for col in df.columns:
        if isinstance(col, str) and re.search(r'giudizio', col, re.IGNORECASE):
            return col
    return None

def find_header_row(file_path, sheet_name):
    """
    Scansiona le prime righe di un foglio di lavoro per identificare la riga
    dell'intestazione che contiene la colonna 'Giudizio'.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        int: L'indice della riga di intestazione o None se non trovata.
    """
    file_path.seek(0)
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 15 righe per trovare l'intestazione
        for i, row in enumerate(sheet.iter_rows(max_row=15)):
            for cell in row:
                if isinstance(cell.value, str) and 'giudizio' in cell.value.lower():
                    return i
        return None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione: {e}")
        return None

def read_excel_file_to_df(file_path, progress_container=None):
    """
    Legge un file Excel e unisce i dati da tutti i fogli di lavoro in un unico DataFrame.
    """
    corpus_list = []
    try:
        file_path.seek(0)
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names
        
        if progress_container is not None:
            progress_container.append(f"Trovati {len(sheet_names)} fogli di lavoro.")
            
        for sheet_name in sheet_names:
            if progress_container is not None:
                progress_container.append(f"Elaborazione del foglio: '{sheet_name}'...")
            
            try:
                # Cerca la riga di intestazione
                file_path.seek(0)
                header_row_index = find_header_row(file_path, sheet_name)
                if header_row_index is None:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Legge il DataFrame con l'intestazione corretta
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)

                # Trova la colonna del giudizio
                giudizio_col = find_giudizio_column(df)
                if giudizio_col is None:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                # Trova la colonna "input"
                input_col = None
                for col in df.columns:
                    if isinstance(col, str) and re.search(r'(input|descrizione|commento|testo)', col, re.IGNORECASE):
                        input_col = col
                        break
                
                if input_col is None:
                    if progress_container is not None:
                        progress_container.append("Colonna 'input' non trovata. La prima colonna verrà usata come prompt.")
                    input_col = df.columns[0]
                
                data_for_dataset = []
                # Itera sul DataFrame e crea un elenco di dizionari
                for _, row in df.iterrows():
                    prompt_text = str(row[input_col]) if pd.notna(row[input_col]) else ""
                    target_text = str(row[giudizio_col]) if pd.notna(row[giudizio_col]) else ""

                    # Aggiunge solo se c'è almeno un prompt valido
                    if prompt_text:
                        data_for_dataset.append({
                            'input_text': prompt_text,
                            'target_text': target_text
                        })
                
                if not data_for_dataset:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                corpus_list.extend(data_for_dataset)
                
            except Exception as e:
                if progress_container is not None:
                    progress_container.append(f"Errore nella lettura del foglio '{sheet_name}': {e}")
                    progress_container.append(traceback.format_exc())
                
        if not corpus_list:
            if progress_container is not None:
                progress_container.append("Nessun dato valido trovato in tutti i fogli del file.")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        if progress_container is not None:
            progress_container.append(f"Errore nella lettura del file: {e}")
            progress_container.append(traceback.format_exc())
        return pd.DataFrame()

def load_and_update_corpus(file_path, progress_container):
    """
    Carica un corpus esistente o ne crea uno nuovo, lo aggiorna con i dati
    del file Excel appena caricato e lo salva.

    Args:
        file_path (BytesIO): Il file Excel caricato.
        progress_container (list): La lista dei messaggi di stato di Streamlit.

    Returns:
        pd.DataFrame: Il DataFrame del corpus aggiornato.
    """
    CORPUS_FILE = "training_corpus.parquet"
    corpus_df = pd.DataFrame()

    try:
        if os.path.exists(CORPUS_FILE):
            progress_container.append("Corpus esistente trovato. Caricamento in corso...")
            corpus_df = pd.read_parquet(CORPUS_FILE)
            progress_container.append(f"Corpus caricato. Totale righe: {len(corpus_df)}")
        else:
            progress_container.append("Nessun corpus esistente trovato. Verrà creato uno nuovo.")

        progress_container.append("Lettura del nuovo file di addestramento...")
        new_df = read_excel_file_to_df(file_path, progress_container)

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
    """
    CORPUS_FILE = "training_corpus.parquet"
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)
        progress_container.append("Corpus di addestramento eliminato con successo.")
    else:
        progress_container.append("Nessun corpus di addestramento da eliminare.")

