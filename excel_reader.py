# ==============================================================================
# File: excel_reader_v2.py
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
from openpyxl.utils.exceptions import InvalidFileException
import warnings

# Ignora i FutureWarnings da openpyxl per una console più pulita
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def make_columns_unique(columns):
    """
    Garantisce che i nomi delle colonne siano unici, aggiungendo un contatore
    se necessario.
    
    Args:
        columns (list): Lista dei nomi delle colonne.
    
    Returns:
        list: Lista dei nomi delle colonne unici.
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
    
    Args:
        df (pd.DataFrame): Il DataFrame grezzo caricato.
        
    Returns:
        tuple: Una tupla contenente l'indice della riga di intestazione, il nome della colonna dei giudizi e il nome della colonna 'pos'.
    """
    try:
        header_row_index = -1
        giudizio_col_name = None
        pos_col_name = None
        
        # Scansiona le prime 50 righe per trovare l'intestazione
        for i in range(min(50, len(df))):
            row_str = " ".join([str(x).lower() for x in df.iloc[i].values if pd.notna(x)])
            if "giudizio" in row_str:
                header_row_index = i
                break
        
        if header_row_index == -1:
            # Fallback se non si trova la parola "giudizio"
            header_row_index = 0
        
        # Imposta la riga di intestazione del DataFrame
        df.columns = make_columns_unique(df.iloc[header_row_index].fillna('').astype(str).str.lower().str.strip().tolist())
        df = df.iloc[header_row_index + 1:].reset_index(drop=True)

        # Trova la colonna 'Giudizio' e 'pos'
        for col in df.columns:
            if re.search(r'giudizio', col, re.IGNORECASE):
                giudizio_col_name = col
            if re.search(r'pos|posizione', col, re.IGNORECASE):
                pos_col_name = col
        
        # Fallback per la colonna 'Giudizio' se non è stata trovata
        if giudizio_col_name is None:
            if len(df.columns) > 7:
                giudizio_col_name = df.columns[7] # Colonna H
            else:
                return -1, None, None # Errore se non si trova la colonna

        return header_row_index, giudizio_col_name, pos_col_name

    except Exception as e:
        return -1, None, None

def trim_dataframe_by_numeric_id_column(df, pos_col_name):
    """
    Tronca il DataFrame in base a una colonna di ID numerici ('pos').
    
    Args:
        df (pd.DataFrame): Il DataFrame da troncare.
        pos_col_name (str): Il nome della colonna di ID numerici.
        
    Returns:
        pd.DataFrame: Il DataFrame troncato.
    """
    if pos_col_name not in df.columns:
        return df

    try:
        df[pos_col_name] = pd.to_numeric(df[pos_col_name], errors='coerce')
        # Trova l'ultima riga dove la sequenza numerica è consecutiva
        last_valid_index = -1
        for i, val in enumerate(df[pos_col_name]):
            if not pd.isna(val) and int(val) == i + 1:
                last_valid_index = i
            else:
                break
        
        if last_valid_index != -1:
            return df.iloc[:last_valid_index + 1]
        else:
            return pd.DataFrame() # Restituisce un DataFrame vuoto se non si trovano dati validi
            
    except Exception:
        return df # Ritorna il DataFrame originale in caso di errore

def trim_dataframe_by_empty_columns(df):
    """
    Rimuove le colonne vuote dalla fine del DataFrame, da destra a sinistra.
    
    Args:
        df (pd.DataFrame): Il DataFrame da pulire.
        
    Returns:
        pd.DataFrame: Il DataFrame con le colonne vuote rimosse.
    """
    # Rimuovi le colonne completamente vuote da destra a sinistra
    cols_to_drop = []
    found_first_non_empty = False
    for col in df.columns[::-1]:
        if df[col].isnull().all() and not found_first_non_empty:
            cols_to_drop.append(col)
        else:
            found_first_non_empty = True
    
    return df.drop(columns=cols_to_drop)

def prepare_data_for_training(df, giudizio_col_name, progress_container):
    """
    Prepara i dati per il fine-tuning del modello.
    
    Args:
        df (pd.DataFrame): Il DataFrame con i dati da preparare.
        giudizio_col_name (str): Il nome della colonna dei giudizi.
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
        
    Returns:
        pd.DataFrame: Il DataFrame preparato per l'addestramento.
    """
    # Rimuovi le righe con giudizio vuoto
    initial_rows = len(df)
    df.dropna(subset=[giudizio_col_name], inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        progress_container(f"Rimosse {rows_dropped} righe con 'Giudizio' mancante.", "warning")

    # Colonne da escludere dal prompt
    exclude_cols_regex = r'pos|posizione|alunno|assenti|cnt'
    
    data_for_dataset = []
    
    for _, row in df.iterrows():
        input_parts = []
        for col in df.columns:
            if col != giudizio_col_name and not re.search(exclude_cols_regex, col, re.IGNORECASE):
                value = row[col]
                if pd.notna(value) and str(value).strip() != '':
                    input_parts.append(f"{col}: {value}")
        
        input_text = " ".join(input_parts)
        target_text = str(row[giudizio_col_name])
        
        if input_text.strip() != '' and target_text.strip() != '':
            data_for_dataset.append({
                'input_text': input_text,
                'target_text': target_text
            })

    return pd.DataFrame(data_for_dataset)

def read_and_prepare_data_from_excel(file_object, sheet_names, progress_container, training_mode=True):
    """
    Legge e prepara i dati da un file Excel, processando più fogli.
    
    Args:
        file_object (BytesIO): Il file Excel caricato.
        sheet_names (list): Lista dei nomi dei fogli da elaborare.
        progress_container (callable): Funzione per inviare messaggi di progresso a Streamlit.
        training_mode (bool): Se True, prepara i dati per l'addestramento. Altrimenti, per la generazione.
        
    Returns:
        pd.DataFrame: Un DataFrame unificato e preparato.
    """
    corpus_list = []
    
    try:
        file_object.seek(0)
        
        # Leggi solo i fogli selezionati per ottimizzare
        all_dfs = pd.read_excel(file_object, sheet_name=sheet_names, header=None)
        
        for sheet in sheet_names:
            df = all_dfs.get(sheet)
            if df is None or df.empty:
                progress_container(f"Attenzione: Foglio '{sheet}' vuoto o non trovato. Saltato.", "warning")
                continue
                
            progress_container(f"Elaborazione del foglio '{sheet}'...", "info")
            
            # Filtra i fogli che non devono essere processati
            if sheet.lower() in ['prototipo', 'medie']:
                progress_container(f"Filtro: Foglio '{sheet}' non viene processato per l'addestramento.", "warning")
                continue
            
            try:
                # Trova l'intestazione e i nomi delle colonne
                header_row_index, giudizio_col_name, pos_col_name = find_header_row_and_columns(df)
                
                if giudizio_col_name is None:
                    progress_container(f"Errore: Colonna 'Giudizio' non trovata nel foglio '{sheet}'. Saltato.", "error")
                    continue
                
                # Applica l'intestazione e pulisci il DataFrame
                df.columns = df.iloc[header_row_index].fillna('').astype(str).str.lower().str.strip().tolist()
                df = df.iloc[header_row_index + 1:].reset_index(drop=True)
                
                # Troncamento basato sulla colonna 'pos'
                df = trim_dataframe_by_numeric_id_column(df, pos_col_name)
                
                # Troncamento delle colonne vuote
                df = trim_dataframe_by_empty_columns(df)
                
                if df.empty:
                    progress_container(f"Attenzione: Nessun dato valido trovato dopo il troncamento nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                # Prepara i dati per l'addestramento o la generazione
                if training_mode:
                    data_for_dataset = prepare_data_for_training(df, giudizio_col_name, progress_container)
                    if not data_for_dataset.empty:
                        corpus_list.extend(data_for_dataset.to_dict('records'))
                else: # Modalità generazione
                    # Rimuovi le righe con giudizio non nullo
                    df_to_process = df[df[giudizio_col_name].isnull()]
                    
                    data_for_dataset = []
                    exclude_cols_regex = r'pos|posizione|alunno|assenti|cnt'
                    
                    for _, row in df_to_process.iterrows():
                        input_parts = []
                        for col in df_to_process.columns:
                            if col != giudizio_col_name and not re.search(exclude_cols_regex, col, re.IGNORECASE):
                                value = row[col]
                                if pd.notna(value) and str(value).strip() != '':
                                    input_parts.append(f"{col}: {value}")
                        
                        input_text = " ".join(input_parts)
                        if input_text.strip() != '':
                            data_for_dataset.append({
                                'input_text': input_text,
                                'original_row': row.to_dict()
                            })
                    
                    corpus_list.extend(data_for_dataset)

            except Exception as e:
                progress_container(f"Errore nella lettura del foglio '{sheet}': {e}", "error")
                progress_container(f"Traceback: {traceback.format_exc()}", "error")
        
        if not corpus_list:
            progress_container("Nessun dato valido trovato in tutti i fogli del file.", "error")
            return pd.DataFrame()
        
        return pd.DataFrame(corpus_list)

    except InvalidFileException:
        progress_container("Errore: Il file caricato non è un file Excel valido.", "error")
        return pd.DataFrame()
    except Exception as e:
        progress_container(f"Errore nella lettura del file: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()


def get_excel_sheet_names(file_object):
    """
    Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
    
    Args:
        file_object (BytesIO): Il file Excel caricato.
        
    Returns:
        list: Lista dei nomi dei fogli.
    """
    try:
        file_object.seek(0) # Riporta il puntatore all'inizio del file
        # 'engine=openpyxl' garantisce la compatibilità con xlsx e xlsm
        workbook = openpyxl.load_workbook(file_object, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except InvalidFileException:
        return []
    except Exception as e:
        return []
