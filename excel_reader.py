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
        int: L'indice della riga di intestazione (0-based) o -1 se non trovata.
    """
    try:
        # Usa openpyxl per la scansione leggera
        wb = openpyxl.load_workbook(file_path, read_only=True)
        ws = wb[sheet_name]
        
        # Scansiona solo le prime 50 righe per efficienza
        for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=50)):
            for cell in row:
                if isinstance(cell.value, str) and re.search(r'giudizio', cell.value, re.IGNORECASE):
                    wb.close()
                    return row_idx
        wb.close()
        return -1
    except Exception as e:
        print(f"Errore nella scansione dell'intestazione: {e}")
        return -1

def read_excel_file_to_df(file_path, progress_container):
    """
    Legge un file Excel da un buffer di byte, trova i dati e li converte in un
    DataFrame di Pandas con le colonne 'input_text' e 'target_text'.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        progress_container (list): Una lista per i messaggi di progresso.

    Returns:
        pd.DataFrame: Un DataFrame con i dati preparati per l'addestramento.
    """
    try:
        file_path.seek(0)
        xlsx = pd.ExcelFile(file_path)
        all_sheets = xlsx.sheet_names
        corpus_list = []

        progress_container.append(f"Lettura di {len(all_sheets)} fogli di lavoro...")

        for sheet_name in all_sheets:
            progress_container.append(f"Elaborazione del foglio: '{sheet_name}'")
            
            # Trova la riga di intestazione
            header_row = find_header_row(file_path, sheet_name)
            
            if header_row == -1:
                progress_container.append(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                continue

            # Legge il DataFrame dal foglio specifico, partendo dalla riga dell'intestazione
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)

            # Trova le colonne necessarie
            giudizio_col = find_giudizio_column(df)
            
            # Elimina le righe con 'Giudizio' vuoto
            if giudizio_col:
                df = df.dropna(subset=[giudizio_col]).copy()
                if df.empty:
                    progress_container.append(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                    continue
            else:
                progress_container.append(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                continue

            # Prepara il DataFrame per l'addestramento
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
                if progress_container is not None:
                    progress_container.append(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                continue

            corpus_list.extend(data_for_dataset)

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

