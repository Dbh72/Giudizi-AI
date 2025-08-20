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
            row = df.iloc[i].astype(str)
            row_lower = row.str.lower()
            
            # Cerca una corrispondenza esatta con "giudizio" o una forma simile
            if 'giudizio' in row_lower.values or 'giudizi' in row_lower.values:
                header_row = df.iloc[i]
                df.columns = make_columns_unique([str(col) for col in header_row])
                df = df.iloc[i+1:].reset_index(drop=True)
                
                giudizio_col = None
                materia_col = None
                desc_col = None
                
                # Trova i nomi delle colonne in modo robusto
                for col in df.columns:
                    if re.match(r"giudizio", str(col).lower()):
                        giudizio_col = col
                    elif re.match(r"materia", str(col).lower()):
                        materia_col = col
                    elif re.match(r"(?:descrizione|note)", str(col).lower(), re.IGNORECASE):
                        desc_col = col
                
                if not all([materia_col, desc_col]):
                    return None, None, None, None, "Intestazioni 'Materia' e/o 'Descrizione' non trovate nel foglio."
                
                return df, giudizio_col, materia_col, desc_col, None
        
        return None, None, None, None, "Intestazione non trovata nelle prime 50 righe."
    except Exception as e:
        return None, None, None, None, f"Errore nella ricerca dell'intestazione: {e}"

def read_excel_for_training(file_object, sheets, progress_container):
    """
    Legge un file Excel per costruire un corpus di addestramento.
    """
    try:
        file_object.seek(0)
        dfs_list = pd.read_excel(file_object, sheet_name=sheets, engine="openpyxl")
        corpus_list = []
        
        for sheet_name, df in dfs_list.items():
            progress_container(f"Elaborazione del foglio '{sheet_name}' per il training...", "info")
            processed_df, giudizio_col, materia_col, desc_col, error_msg = find_header_row_and_columns(df)
            
            if error_msg:
                progress_container(f"Errore nel foglio '{sheet_name}': {error_msg}", "error")
                continue
                
            if not all([giudizio_col, materia_col, desc_col]):
                progress_container(f"Avviso: Le colonne 'Giudizio', 'Materia' e/o 'Descrizione' non sono state trovate nel foglio '{sheet_name}'. Il foglio verrà saltato.", "warning")
                continue
            
            # Filtra le righe con valori mancanti nelle colonne chiave
            processed_df.dropna(subset=[giudizio_col, materia_col, desc_col], inplace=True)
            processed_df = processed_df[processed_df[giudizio_col].astype(str).str.strip() != '']
            processed_df = processed_df[processed_df[materia_col].astype(str).str.strip() != '']
            processed_df = processed_df[processed_df[desc_col].astype(str).str.strip() != '']
            
            if processed_df.empty:
                progress_container(f"Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.", "warning")
                continue
            
            # Costruisci il DataFrame del corpus
            data_for_dataset = [{
                'prompt': f"Materia: {row[materia_col]} - Descrizione Giudizio: {row[desc_col]}",
                'target_text': str(row[giudizio_col])
            } for index, row in processed_df.iterrows()]
            
            corpus_list.extend(data_for_dataset)
        
        if not corpus_list:
            progress_container("Nessun dato valido trovato in tutti i fogli del file per il training.", "error")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        progress_container(f"Errore nella lettura del file per il training: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()


def read_excel_for_generation(file_object, sheet_name, progress_container):
    """
    Legge un file Excel per la generazione dei giudizi.
    """
    try:
        file_object.seek(0)
        df = pd.read_excel(file_object, sheet_name=sheet_name, engine="openpyxl")
        
        processed_df, giudizio_col, materia_col, desc_col, error_msg = find_header_row_and_columns(df)
        
        if error_msg:
            progress_container(f"Errore: {error_msg}", "error")
            return pd.DataFrame()

        if processed_df is None:
            progress_container("Impossibile trovare le colonne necessarie nel file.", "error")
            return pd.DataFrame()
        
        progress_container(f"Trovate le colonne: Giudizio='{giudizio_col}', Materia='{materia_col}', Descrizione='{desc_col}'", "info")
        
        # Gestisci il caso in cui la colonna 'Giudizio' è vuota o inesistente
        if giudizio_col not in processed_df.columns:
            processed_df[giudizio_col] = ''
        
        return processed_df, giudizio_col, materia_col, desc_col

    except Exception as e:
        progress_container(f"Errore nella lettura del file per la generazione: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame(), None, None, None

def get_excel_sheet_names(file_object):
    """
    Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
    """
    try:
        file_object.seek(0) # Riporta il puntatore all'inizio del file
        workbook = openpyxl.load_workbook(file_object, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except Exception as e:
        return [f"Errore: {e}"]
