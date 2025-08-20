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
    Implementa la ricerca flessibile nelle prime 10 righe.
    """
    try:
        # Cerca la riga di intestazione nelle prime 10 righe
        for i in range(min(10, len(df))):
            row = df.iloc[i].astype(str).str.lower()
            if any("giudizio" in str(cell) for cell in row):
                df.columns = df.iloc[i]
                df = df.iloc[i+1:].reset_index(drop=True)
                break
        else:
            # Fallback: se non trova "giudizio", usa la prima riga come header
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)

        # Rendi i nomi delle colonne unici
        df.columns = make_columns_unique(df.columns.astype(str))

        giudizio_col_name = None
        pos_col_name = None
        
        # Cerca la colonna 'Giudizio' in modo flessibile
        for col in df.columns:
            cleaned_col = re.sub(r'\s+', '', col).lower()
            if cleaned_col == "giudizio":
                giudizio_col_name = col
                break
        
        # Fallback a colonna H (indice 7) se non trovata
        if giudizio_col_name is None and len(df.columns) > 7:
            giudizio_col_name = df.columns[7]

        # Cerca la colonna 'pos' in modo flessibile
        for col in df.columns:
            cleaned_col = re.sub(r'\s+', '', col).lower()
            if cleaned_col == "pos":
                pos_col_name = col
                break
        
        return df, giudizio_col_name, pos_col_name

    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione: {e}")
        return pd.DataFrame(), None, None

def trim_dataframe_by_numeric_id_column(df, pos_col_name):
    """
    Tronca il DataFrame basandosi sulla colonna 'pos'.
    La lettura si interrompe quando la sequenza numerica si spezza o termina.
    """
    if pos_col_name is None or pos_col_name not in df.columns:
        return df

    last_valid_index = -1
    for i, value in enumerate(df[pos_col_name]):
        try:
            if not pd.isna(value):
                # Assicurati che il valore sia un numero e che la sequenza continui
                current_pos = int(value)
                if last_valid_index != -1 and current_pos != last_valid_index + 1:
                    break
                last_valid_index = current_pos
            else:
                # Se il valore è vuoto, considera la sequenza interrotta
                break
        except (ValueError, TypeError):
            # Se il valore non è un numero, considera la sequenza interrotta
            break
            
    if last_valid_index == -1:
        return pd.DataFrame()
    
    return df.iloc[:last_valid_index].reset_index(drop=True)


def trim_dataframe_by_empty_columns(df):
    """
    Tronca le colonne vuote dal lato destro del DataFrame.
    La rimozione si ferma quando si trovano due colonne consecutive non vuote.
    """
    if df.empty:
        return df

    # Rimuovi le colonne che sono completamente vuote
    df.dropna(axis=1, how='all', inplace=True)
    return df

def read_and_prepare_data_from_excel(file_object, sheets_to_read, progress_container):
    """
    Legge e prepara i dati da uno o più fogli di un file Excel.
    Implementa la logica di troncamento, pulizia e creazione del prompt.
    """
    try:
        file_object.seek(0)
        
        corpus_list = []
        
        for sheet in sheets_to_read:
            progress_container(f"Elaborazione del foglio '{sheet}'...", "info")
            
            try:
                # Leggi il foglio senza intestazioni per trovarle in modo flessibile
                df = pd.read_excel(file_object, sheet_name=sheet, header=None)
                
                # Trova la riga di intestazione e le colonne chiave
                df, giudizio_col_name, pos_col_name = find_header_row_and_columns(df)
                
                if df.empty or giudizio_col_name is None:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}' o colonna 'Giudizio' mancante. Saltato.", "warning")
                    continue
                
                # Troncamento delle righe basato sulla colonna 'pos'
                if pos_col_name:
                    df = trim_dataframe_by_numeric_id_column(df, pos_col_name)

                # Troncamento delle colonne vuote
                df = trim_dataframe_by_empty_columns(df)
                
                # Rimuovi le righe che sono diventate completamente vuote
                df.dropna(how='all', inplace=True)
                
                # Rimuovi le righe dove il giudizio è vuoto
                df.dropna(subset=[giudizio_col_name], inplace=True)
                
                # Identificazione delle colonne da escludere
                excluded_cols = ['alunno', 'assenti', 'cnt', 'pos']
                
                # Creazione del dataset per il fine-tuning
                data_for_dataset = []
                for _, row in df.iterrows():
                    input_parts = []
                    for col in df.columns:
                        # Pulisci il nome della colonna per il controllo di esclusione
                        cleaned_col = re.sub(r'\s+', '', str(col)).lower()
                        
                        # Se la colonna non è da escludere e non è il giudizio
                        if cleaned_col not in excluded_cols and col != giudizio_col_name:
                            value = str(row[col]).strip()
                            if value:
                                input_parts.append(f"{col}: {value}")

                    input_text = " ".join(input_parts)
                    target_text = str(row[giudizio_col_name]).strip()
                    
                    if input_text and target_text:
                        data_for_dataset.append({
                            'input_text': input_text,
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
        print(f"Errore nella lettura dei nomi dei fogli: {e}")
        return []

