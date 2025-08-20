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
            if any(re.search(r'descrizion(?:e|i)?', x) for x in row):
                desc_col_name = None
                giudizio_col_name = None
                
                # Cerca le colonne 'descrizione' e 'giudizio'
                for col in row.index:
                    if re.search(r'descrizion(?:e|i)?', row[col]):
                        desc_col_name = col
                    if re.search(r'giudizio', row[col]):
                        giudizio_col_name = col
                
                if desc_col_name:
                    return i, desc_col_name, giudizio_col_name
        return None, None, None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione: {e}")
        return None, None, None

# ==============================================================================
# SEZIONE 2: FUNZIONI PRINCIPALI
# ==============================================================================

def read_excel_file(file_object, selected_sheet, progress_container):
    """
    Legge un file Excel, identifica le colonne, estrae i dati rilevanti
    e restituisce un DataFrame standardizzato.
    """
    try:
        file_object.seek(0)
        
        # Legge tutti i dati del foglio selezionato
        df = pd.read_excel(file_object, sheet_name=selected_sheet, header=None, engine='openpyxl')
        
        # Trova la riga di intestazione e le colonne
        header_row_index, desc_col_name, giudizio_col_name = find_header_row_and_columns(df)
        
        if header_row_index is None:
            progress_container("Attenzione: Non è stata trovata una riga di intestazione valida con la colonna 'Descrizione'. Assicurati che il file contenga questa intestazione.", "warning")
            return pd.DataFrame()

        # Imposta le intestazioni del DataFrame
        df.columns = make_columns_unique(df.iloc[header_row_index])
        
        # Rimuove le righe prima dell'intestazione e resetta l'indice
        df = df[header_row_index+1:].reset_index(drop=True)
        
        # Rinomina le colonne
        df.rename(columns={
            desc_col_name: "Descrizione",
            giudizio_col_name: "Giudizio"
        }, inplace=True)
        
        # Se la colonna 'Giudizio' non è stata trovata, la aggiunge
        if 'Giudizio' not in df.columns:
            df['Giudizio'] = ""
            progress_container("Attenzione: Colonna 'Giudizio' non trovata. Verrà creata vuota.", "warning")

        progress_container("File letto e dati preparati con successo.", "success")
        return df

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
        print(f"Errore nella lettura dei nomi dei fogli: {e}")
        return []
