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
        int: L'indice di riga (0-based) dell'intestazione o None se non trovata.
    """
    file_path.seek(0)
    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = workbook[sheet_name]
    
    for row_idx, row in enumerate(sheet.iter_rows(max_row=50)):
        for cell in row:
            if isinstance(cell.value, str) and re.search(r'giudizio', cell.value, re.IGNORECASE):
                workbook.close()
                return row_idx
    
    workbook.close()
    return None

def read_excel_file_to_df(file_path, progress_container):
    """
    Legge un file Excel da un oggetto BytesIO in un DataFrame pandas.
    Seleziona tutti i fogli di lavoro e cerca la colonna 'Giudizio'.
    Ignora i fogli che non contengono la colonna.

    Args:
        file_path (BytesIO): L'oggetto BytesIO del file Excel.
        progress_container (callable): Funzione per inviare messaggi di stato.
        
    Returns:
        pd.DataFrame: Un DataFrame combinato di tutti i dati validi, o un
                      DataFrame vuoto in caso di errore.
    """
    try:
        corpus_list = []
        
        # Legge tutti i fogli del file
        file_path.seek(0)
        all_sheets_df = pd.read_excel(file_path, sheet_name=None)
        
        for sheet in all_sheets_df:
            try:
                progress_container(f"Analisi del foglio: '{sheet}'...", "info")
                df = all_sheets_df[sheet]
                
                # Trova la colonna 'Giudizio'
                giudizio_col = find_giudizio_column(df)
                
                if not giudizio_col:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Questo foglio verrà saltato.", "warning")
                    continue
                
                # Rimuove le righe dove il giudizio è vuoto o non è una stringa
                df = df[df[giudizio_col].apply(lambda x: isinstance(x, str) and x.strip() != '')]
                
                if df.empty:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}'. Questo foglio verrà saltato.", "warning")
                    continue
                    
                # Costruisce il dataset per l'addestramento
                data_for_dataset = []
                for index, row in df.iterrows():
                    # Rimuove la colonna 'Giudizio' per creare l'input_text
                    input_data = row.drop(labels=[giudizio_col])
                    prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val)])
                    target_text = str(row[giudizio_col]) if pd.notna(row[giudizio_col]) else ""

                    # Aggiunge solo se c'è almeno un prompt valido
                    if prompt_text:
                        data_for_dataset.append({
                            'input_text': prompt_text,
                            'target_text': target_text
                        })

                if not data_for_dataset:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                corpus_list.extend(data_for_dataset)
            
            except Exception as e:
                progress_container(f"Errore nella lettura del foglio '{sheet}': {e}", "error")
                progress_container(f"Traceback: {traceback.format_exc()}", "error")
        
        if not corpus_list:
            progress_container("Nessun dato valido trovato in tutti i fogli del file.", "error")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        progress_container(f"Errore nella lettura del file: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()

