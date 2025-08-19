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
            row = df.iloc[i].astype(str).str.lower()
            if 'giudizio' in row.values:
                header_row = df.iloc[i]
                df.columns = make_columns_unique(header_row.values)
                df = df.iloc[i+1:].reset_index(drop=True)
                giudizio_col = next((col for col in df.columns if isinstance(col, str) and 'giudizio' in col.lower()), None)
                return df, giudizio_col
        return df, None
    except Exception as e:
        print(f"Errore nella ricerca dell'header: {e}")
        return df, None

def read_and_prepare_data_from_excel(file_object, sheet_names, progress_container):
    """
    Legge un file Excel, ne estrae i dati, crea un corpus di addestramento
    e lo restituisce come DataFrame.
    """
    corpus_list = []
    
    try:
        progress_container(f"Lettura del file Excel: {file_object.name}", "info")
        
        for sheet in sheet_names:
            progress_container(f"Elaborazione del foglio: '{sheet}'...", "info")
            if "prototipo" in sheet.lower() or "medie" in sheet.lower():
                progress_container(f"Ignorando il foglio '{sheet}' (prototipo o medie). Saltato.", "warning")
                continue
            
            try:
                file_object.seek(0) # Riporta il puntatore all'inizio del file per ogni foglio
                df = pd.read_excel(file_object, sheet_name=sheet, header=None)
                df, giudizio_col = find_header_row_and_columns(df)

                if giudizio_col is None:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                # Prepara i dati per il dataset
                data_for_dataset = []
                for _, row in df.iterrows():
                    input_data = row.drop(labels=[c for c in df.columns if isinstance(c, str) and ('giudizio' in c.lower() or 'alunno' in c.lower() or 'assenti' in c.lower() or 'cnt' in c.lower() or 'pos' in c.lower())], errors='ignore')
                    prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])
                    target_text = str(row[giudizio_col]) if pd.notna(row[giudizio_col]) else ""

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
        print(f"Errore nel recupero dei nomi dei fogli: {e}")
        return []

