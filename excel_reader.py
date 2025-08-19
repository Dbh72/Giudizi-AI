# ==============================================================================
# File: excel_reader.py
# Logica per la preparazione dei dati da file Excel.
# Include tutte le funzioni necessarie per la lettura dei fogli e l'estrazione dei dati.
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
    Trova la riga di intestazione e la colonna 'Giudizio' in un DataFrame.
    """
    try:
        for i in range(min(50, len(df))):
            row = df.iloc[i].astype(str)
            header_found = False
            giudizio_col = -1
            
            for j, col_name in enumerate(row):
                if re.search(r'giudizio', str(col_name).lower()):
                    giudizio_col = j
                    header_found = True
                    break
            
            if header_found:
                df.columns = make_columns_unique(df.iloc[i].astype(str))
                giudizio_col_name = df.columns[giudizio_col]
                df = df.iloc[i+1:].reset_index(drop=True)
                return df, giudizio_col_name
        
        return None, None
    except Exception as e:
        return None, None

def read_and_prepare_data_from_excel(file_path, sheet_names, progress_container):
    """
    Legge un file Excel, ne estrae i dati, crea un corpus di addestramento
    e lo restituisce come DataFrame.
    """
    corpus_list = []
    
    try:
        progress_container(f"Lettura del file Excel: {file_path}", "info")
        
        for sheet in sheet_names:
            progress_container(f"Elaborazione del foglio: '{sheet}'...", "info")
            if "prototipo" in sheet.lower() or "medie" in sheet.lower():
                progress_container(f"Ignorando il foglio '{sheet}' (prototipo o medie).", "warning")
                continue
            
            try:
                df = pd.read_excel(file_path, sheet_name=sheet, header=None)
                df, giudizio_col = find_header_row_and_columns(df)
                
                if giudizio_col is None:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Saltato.", "warning")
                    continue

                ignored_columns = ['alunno', 'assenti', 'cnt', 'pos']
                
                input_columns = [
                    col for col in df.columns 
                    if not any(re.search(word, str(col).lower()) for word in ignored_columns) and col != giudizio_col
                ]

                data_for_dataset = []
                for index, row in df.iterrows():
                    if pd.isna(row[giudizio_col]) or str(row[giudizio_col]).strip() == "":
                        continue
                    
                    input_data = row[input_columns].astype(str).to_dict()
                    prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val) and str(val).strip() != ''])
                    target_text = str(row[giudizio_col])

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


def get_excel_sheet_names(file_path):
    """
    Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except Exception as e:
        print(f"Errore nel recupero dei nomi dei fogli: {e}")
        return []

