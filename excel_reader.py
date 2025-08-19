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
        int: L'indice di riga (0-based) dell'intestazione, o -1 se non trovata.
    """
    workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = workbook[sheet_name]
    
    # Legge le prime 100 righe per trovare l'intestazione
    for row_idx, row in enumerate(sheet.iter_rows(max_row=100)):
        for cell in row:
            if isinstance(cell.value, str) and re.search(r'giudizio', cell.value, re.IGNORECASE):
                workbook.close()
                return row_idx
    
    workbook.close()
    return -1

def get_excel_sheet_names(file_path, progress_container):
    """
    Legge i nomi dei fogli di lavoro da un file Excel.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        progress_container (callable): Funzione per inviare messaggi di stato.

    Returns:
        list: Una lista di nomi dei fogli di lavoro.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except Exception as e:
        progress_container(f"Errore nella lettura dei fogli di lavoro: {e}")
        return []

def read_excel_file_to_df(file_path, progress_container, sheet_name=None, read_only=False):
    """
    Legge i dati da un file Excel e li converte in un DataFrame, cercando l'intestazione
    corretta e la colonna 'Giudizio'.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        progress_container (callable): Funzione per inviare messaggi di stato.
        sheet_name (str, optional): Il nome specifico del foglio da leggere.
        read_only (bool, optional): Se True, elabora solo le righe con colonna 'Giudizio' vuota.

    Returns:
        pd.DataFrame: Un DataFrame contenente 'input_text' e 'target_text'
                      o un DataFrame vuoto se fallisce.
    """
    try:
        if sheet_name:
            sheets_to_process = [sheet_name]
        else:
            sheets_to_process = get_excel_sheet_names(file_path, progress_container)
            if not sheets_to_process:
                progress_container("Nessun foglio di lavoro trovato.", "error")
                return pd.DataFrame()
        
        corpus_list = []
        for sheet in sheets_to_process:
            try:
                # Trova la riga dell'intestazione per il foglio corrente
                header_row_index = find_header_row(file_path, sheet)
                if header_row_index == -1:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Questo foglio verrà saltato.", "warning")
                    continue
                
                # Legge il foglio in un DataFrame a partire dalla riga dell'intestazione
                df = pd.read_excel(file_path, sheet_name=sheet, header=header_row_index, engine='openpyxl')
                
                # Trova di nuovo il nome esatto della colonna 'Giudizio'
                giudizio_col = find_giudizio_column(df)
                if not giudizio_col:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Questo foglio verrà saltato.", "warning")
                    continue
                
                # Logica per elaborare solo le righe con 'Giudizio' vuoto in modalità read_only
                if read_only:
                    df = df[df[giudizio_col].isna()]
                    if df.empty:
                        progress_container(f"Attenzione: Nessuna riga da completare trovata nel foglio '{sheet}'.", "warning")
                        continue

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
