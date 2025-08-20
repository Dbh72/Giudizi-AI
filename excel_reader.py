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
            giudizio_col = [col for col in row if 'giudizio' in col.lower()]
            if giudizio_col:
                header_row = i
                df.columns = make_columns_unique(df.iloc[header_row].astype(str))
                df = df.iloc[header_row:].reset_index(drop=True)
                giudizio_col_name = giudizio_col[0]

                # Tenta di trovare anche le colonne di input in base ai nomi
                input_cols = [col for col in df.columns if 'input' in col.lower() or 'testo' in col.lower() or 'descrizione' in col.lower()]
                if not input_cols:
                    input_cols = [col for col in df.columns if col != giudizio_col_name]
                    
                input_cols_name = input_cols if input_cols else [None]

                return header_row, giudizio_col_name, input_cols_name
    except Exception as e:
        # Questo errore non dovrebbe bloccare la ricerca, quindi lo stampiamo e continuiamo
        # progress_container(f"Errore nella ricerca dell'header: {e}", "warning")
        pass

    return None, None, None

def read_excel_file(file_object, sheet_name, progress_container):
    """
    Legge un file Excel e restituisce un DataFrame di pandas, pulendo i dati.
    """
    try:
        # Riporta il puntatore all'inizio del file per evitare problemi
        file_object.seek(0)
        
        # Carica il file, specificando solo il foglio richiesto
        df = pd.read_excel(file_object, sheet_name=sheet_name, header=None)
        
        # Trova la riga di intestazione e le colonne
        header_row, giudizio_col_name, input_cols_name = find_header_row_and_columns(df.copy())
        
        if giudizio_col_name is None:
            progress_container(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet_name}'. Assicurati che il nome sia corretto (es. 'Giudizio', 'giudizio finale', ecc.).", "error")
            return pd.DataFrame()

        # Ricarica il DataFrame con la riga di intestazione corretta
        df = pd.read_excel(file_object, sheet_name=sheet_name, header=header_row)
        
        df.columns = make_columns_unique(df.columns.astype(str))
        
        giudizio_col_name_unique = [c for c in df.columns if 'giudizio' in c.lower() and c.lower() == giudizio_col_name.lower()][0]
        
        input_cols_name_unique = [c for c in df.columns if c in input_cols_name]
        
        if not input_cols_name_unique:
            progress_container(f"Errore: Nessuna colonna di input (es. 'Testo', 'Descrizione') trovata nel foglio '{sheet_name}'.", "error")
            return pd.DataFrame()

        # Se il file ha il formato corretto, lo restituiamo
        df_valid = df.dropna(subset=[input_cols_name_unique[0]]).copy()
        
        return df_valid, giudizio_col_name_unique, input_cols_name_unique

    except Exception as e:
        progress_container(f"Errore nella lettura del file: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame(), None, None

def read_training_file(file_object, sheet_names, progress_container):
    """
    Legge un file Excel con più fogli e restituisce un DataFrame di pandas
    per l'addestramento.
    """
    try:
        corpus_list = []
        file_object.seek(0)
        
        for sheet in sheet_names:
            try:
                df = pd.read_excel(file_object, sheet_name=sheet, header=None)
                header_row, giudizio_col_name, input_cols_name = find_header_row_and_columns(df.copy())
                
                if giudizio_col_name is None:
                    progress_container(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                    
                df_data = pd.read_excel(file_object, sheet_name=sheet, header=header_row)
                df_data.columns = make_columns_unique(df_data.columns.astype(str))
                
                giudizio_col_name_unique = [c for c in df_data.columns if 'giudizio' in c.lower() and c.lower() == giudizio_col_name.lower()][0]
                input_cols_name_unique = [c for c in df_data.columns if c in input_cols_name]
                
                df_data = df_data.dropna(subset=[giudizio_col_name_unique] + input_cols_name_unique).copy()

                data_for_dataset = []
                for _, row in df_data.iterrows():
                    source_text_parts = [str(row[col]) for col in input_cols_name_unique]
                    source_text = " ".join(source_text_parts)
                    target_text = str(row[giudizio_col_name_unique])
                    
                    data_for_dataset.append({
                        'input_text': source_text,
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
        st.error(f"Errore nella lettura dei fogli del file: {e}")
        return []

