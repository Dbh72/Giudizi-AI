# ==============================================================================
# File: excel_reader.py
# Logica per la preparazione dei dati da file Excel.
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
            row = df.iloc[i].astype(str).str.strip().str.lower()
            if 'giudizio' in row.tolist():
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                df.columns = make_columns_unique(df.columns)
                giudizio_col = next((col for col in df.columns if 'giudizio' in str(col).lower()), None)
                if giudizio_col:
                    return df, giudizio_col
        return None, None
    except Exception:
        return None, None

def read_and_prepare_data_from_excel(uploaded_file, progress_container):
    """
    Legge un file Excel, identifica le colonne, estrae i dati di input e target
    e restituisce un DataFrame pronto per l'addestramento.
    """
    try:
        progress_container("Lettura del file Excel per l'addestramento...", "info")
        
        file_bytes = BytesIO(uploaded_file.getvalue())
        excel_data = pd.ExcelFile(file_bytes)
        
        corpus_list = []
        
        for sheet_name in excel_data.sheet_names:
            if sheet_name.lower() in ['prototipo', 'medie']:
                progress_container(f"Saltato il foglio '{sheet_name}'.", "warning")
                continue
            
            progress_container(f"Processo del foglio '{sheet_name}'...", "info")
            df = pd.read_excel(excel_data, sheet_name=sheet_name, header=None)
            
            df_cleaned, giudizio_col = find_header_row_and_columns(df)
            
            if df_cleaned is None or giudizio_col is None:
                progress_container(f"Attenzione: Non è stata trovata una colonna 'Giudizio' nel foglio '{sheet_name}'. Saltato.", "warning")
                continue
            
            data_for_dataset = []
            
            for index, row in df_cleaned.iterrows():
                input_data = row.drop(giudizio_col, errors='ignore')
                
                # Esclude colonne specifiche (case-insensitive)
                exclude_cols = ['alunno', 'assenti', 'cnt', 'pos']
                input_data = input_data.loc[[col for col in input_data.index if str(col).lower() not in exclude_cols]]

                # Genera il prompt di input
                prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])
                target_text = str(row[giudizio_col]) if pd.notna(row[giudizio_col]) else ""

                if prompt_text and target_text:
                    data_for_dataset.append({
                        'input_text': prompt_text,
                        'target_text': target_text
                    })
            
            if not data_for_dataset:
                progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.", "warning")
                continue
            
            corpus_list.extend(data_for_dataset)
        
        if not corpus_list:
            progress_container("Nessun dato valido trovato in tutti i foghi del file.", "error")
            return pd.DataFrame()
        
        progress_container(f"Trovate {len(corpus_list)} righe di dati valide per l'addestramento.", "success")
        return pd.DataFrame(corpus_list)

    except Exception as e:
        progress_container(f"Errore nella lettura del file: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()

def get_excel_sheet_names(uploaded_file):
    """
    Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
    """
    try:
        file_bytes = BytesIO(uploaded_file.getvalue())
        workbook = openpyxl.load_workbook(file_bytes, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except Exception as e:
        return [f"Errore: {e}"]
