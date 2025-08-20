# ==============================================================================
# File: excel_reader.py
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

def find_header_row_and_columns(df, progress_container):
    """
    Trova la riga di intestazione e le posizioni della colonna 'Giudizio'.
    """
    try:
        giudizio_col_name = None
        header_row_index = -1
        
        # Scansiona le prime 10 righe per trovare l'intestazione
        for i in range(min(10, len(df))):
            row = df.iloc[i].astype(str).str.lower()
            if any(re.search(r'\b(giudizio|giudiz)\b', str(cell)) for cell in row):
                header_row_index = i
                break
        
        # Se non trova una riga di intestazione, usa la prima riga
        if header_row_index == -1:
            progress_container("Riga di intestazione non trovata. Supponendo che sia la prima riga.", "warning")
            header_row_index = 0

        df.columns = df.iloc[header_row_index]
        df = df.iloc[header_row_index + 1:].reset_index(drop=True)

        # Rende i nomi delle colonne unici
        df.columns = make_columns_unique(df.columns.tolist())

        # Cerca la colonna "Giudizio" in modo flessibile
        for col in df.columns:
            if re.search(r'\b(giudizio|giudiz)\b', str(col).lower()):
                giudizio_col_name = col
                progress_container(f"Colonna 'Giudizio' trovata: '{giudizio_col_name}'", "info")
                break
        
        # Fallback sulla colonna H (indice 7) se non trovata
        if giudizio_col_name is None:
            if len(df.columns) > 7:
                giudizio_col_name = df.columns[7]
                progress_container(f"Colonna 'Giudizio' non trovata. Utilizzo la colonna H ('{giudizio_col_name}') come fallback.", "warning")
            else:
                progress_container("Impossibile trovare la colonna 'Giudizio' e la colonna H non esiste.", "error")
                return None, None

        return df, giudizio_col_name

    except Exception as e:
        progress_container(f"Errore nella ricerca dell'intestazione: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None

def trim_dataframe_by_empty_columns(df):
    """
    Rimuove le colonne vuote consecutive dalla destra.
    """
    empty_cols = 0
    cols_to_keep = len(df.columns)
    for i in range(len(df.columns) - 1, -1, -1):
        if df.iloc[:, i].isnull().all():
            empty_cols += 1
        else:
            break
        
    return df.iloc[:, :cols_to_keep - empty_cols]

def trim_dataframe_by_numeric_id_column(df):
    """
    Tronca il DataFrame quando la sequenza numerica in una colonna di ID si interrompe.
    La colonna di ID viene identificata in modo flessibile.
    """
    id_col = None
    # Identifica la colonna 'pos' o simili
    for col in df.columns:
        if re.search(r'\b(pos|id|numero)\b', str(col).lower()):
            id_col = col
            break
            
    if id_col is None:
        return df

    # Converte la colonna in tipo numerico, gestendo gli errori
    df[id_col] = pd.to_numeric(df[id_col], errors='coerce')
    
    last_valid_index = -1
    for i in range(len(df)):
        if not pd.isna(df.loc[i, id_col]) and isinstance(df.loc[i, id_col], (int, float)):
            last_valid_index = i
        else:
            break
            
    if last_valid_index != -1:
        return df.iloc[:last_valid_index + 1]
    
    return df

def read_and_prepare_data_from_excel(file_object, sheet_names, progress_container):
    """
    Legge e prepara i dati da un file Excel, processando tutti i fogli
    specificati e unendoli.
    """
    try:
        corpus_list = []
        # FILTRO CORRETTO: la logica per ignorare i fogli è ora qui,
        # senza dipendere da parametri esterni.
        sheets_to_ignore = ['prototipo', 'medie']
        
        for sheet in sheet_names:
            if sheet.lower().strip() in [sh.lower().strip() for sh in sheets_to_ignore]:
                progress_container(f"Fogli '{sheet}' con 'prototipo' o 'medie' nel nome verranno ignorati.", "warning")
                continue

            try:
                progress_container(f"Lettura del foglio: {sheet}...", "info")
                file_object.seek(0)
                # Utilizza l'argomento 'engine' per gestire diversi formati
                df = pd.read_excel(file_object, sheet_name=sheet, header=None, engine='openpyxl')
                
                # Trova l'intestazione e la colonna 'Giudizio'
                df, giudizio_col_name = find_header_row_and_columns(df, progress_container)
                
                if df is None:
                    continue
                
                # Troncamento delle righe basato sulla colonna 'pos'
                df = trim_dataframe_by_numeric_id_column(df)

                # Troncamento delle colonne vuote
                df = trim_dataframe_by_empty_columns(df)

                # Rimuove righe che sono diventate vuote dopo il troncamento
                df.dropna(how='all', inplace=True)

                if giudizio_col_name is None:
                    progress_container(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Saltato.", "warning")
                    continue

                # Rimuove le righe con il giudizio vuoto, per assicurare che il corpus sia valido
                df.dropna(subset=[giudizio_col_name], inplace=True)
                
                # Pulisce i dati e crea il prompt
                data_for_dataset = []
                for _, row in df.iterrows():
                    target_text = str(row[giudizio_col_name]).strip()
                    if pd.isna(target_text) or not target_text:
                        continue
                    
                    # Costruisce il prompt
                    prompt_parts = []
                    for col in df.columns:
                        if (re.search(r'\b(alunno|assenti|cnt|pos)\b', str(col).lower())
                            or str(col) == giudizio_col_name):
                            continue
                        
                        cell_value = str(row[col]).strip()
                        if pd.isna(cell_value) or not cell_value:
                            continue
                        
                        prompt_parts.append(f"{str(col).strip()}: {cell_value}")
                    
                    if prompt_parts:
                        input_text = " ".join(prompt_parts)
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
        st.error(f"Errore nel leggere i nomi dei fogli: {e}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return []
