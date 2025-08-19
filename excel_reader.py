# ==============================================================================
# File: excel_reader_v2.py
# Logica per la preparazione dei dati da file Excel, integrando le
# funzionalità di '33 Funziona.txt' per una lettura più robusta.
# ==============================================================================
import pandas as pd
import openpyxl
import re
from io import BytesIO
import traceback
import os
import shutil
import json
from datetime import datetime

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def make_columns_unique(columns):
    """
    Garantisce che i nomi delle colonne siano unici, aggiungendo un contatore
    se necessario.
    """
    seen = {}
    new_columns = []
    for col in columns:
        original_col = col
        if original_col in seen:
            seen[original_col] += 1
            new_columns.append(f"{original_col}_{seen[original_col]}")
        else:
            seen[original_col] = 0
            new_columns.append(original_col)
    return new_columns

def find_header_row_and_columns(df):
    """
    Trova la riga di intestazione e le posizioni della colonna 'Giudizio'.
    """
    try:
        for i in range(min(50, len(df))):
            row_values = df.iloc[i].astype(str).str.lower()
            try:
                # Cerca la colonna 'Giudizio'
                giudizio_col_found_idx = next(idx for idx, val in enumerate(row_values) if 'giudizio' in val.strip())
                
                if giudizio_col_found_idx is not None:
                    header_row = df.iloc[i].ffill().str.strip()
                    giudizio_col_name = header_row.iloc[giudizio_col_found_idx]
                    return i, {'Giudizio': giudizio_col_name}
            except StopIteration:
                continue
        
        raise ValueError("Non è stato possibile trovare una riga di intestazione con la colonna 'Giudizio' nelle prime 50 righe.")
    except Exception as e:
        print(f"ERRORE in find_header_row_and_columns: {e}")
        raise e

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
        
        file_path.seek(0)
        all_sheets_df = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        
        for sheet in all_sheets_df:
            try:
                progress_container(f"Analisi del foglio: '{sheet}'...", "info")
                df_original = all_sheets_df[sheet]
                
                # Trova la riga di intestazione e le colonne necessarie
                header_row_index, column_mapping = find_header_row_and_columns(df_original.copy())

                clean_columns = df_original.iloc[header_row_index].ffill().str.strip().tolist()
                unique_columns = make_columns_unique(clean_columns)
                
                df = df_original.iloc[header_row_index + 1:].copy()
                df.columns = unique_columns
                
                giudizio_col = column_mapping['Giudizio']
                
                # Rimuove le righe dove il giudizio è vuoto o non è una stringa
                df = df[df[giudizio_col].apply(lambda x: isinstance(x, str) and x.strip() != '')]
                
                if df.empty:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}'. Questo foglio verrà saltato.", "warning")
                    continue
                    
                # Costruisce il dataset per l'addestramento
                data_for_dataset = []
                for index, row in df.iterrows():
                    # Rimuove la colonna 'Giudizio' per creare l'input_text
                    input_data = row.drop(labels=[giudizio_col], errors='ignore')
                    prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])
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
