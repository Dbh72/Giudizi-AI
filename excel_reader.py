# ==============================================================================
# File: excel_reader.py
# Logica per la preparazione dei dati da file Excel
# ==============================================================================
import pandas as pd
import openpyxl
import re
from io import BytesIO

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def find_giudizio_column(df):
    """
    Trova la colonna 'Giudizio' nel DataFrame, cercando in modo case-insensitive
    in tutte le intestazioni.

    Args:
        df (pd.DataFrame): Il DataFrame del foglio da analizzare.

    Returns:
        str: Il nome della colonna 'Giudizio' o None se non trovata.
    """
    # Cerca la parola 'giudizio' in modo case-insensitive tra le colonne.
    for col in df.columns:
        if isinstance(col, str) and re.search(r'giudizio', col, re.IGNORECASE):
            return col
    return None

def find_header_row(file_path, sheet_name):
    """
    Scansiona le prime righe di un foglio di lavoro per identificare la riga
    dell'intestazione che contiene la colonna 'Giudizio'.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        int: L'indice di riga dell'intestazione (basato su 0) o None se non trovata.
    """
    try:
        # Carica il file in modalità di sola lettura per trovare l'intestazione
        file_path.seek(0)
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook[sheet_name]
        
        # Scansione delle prime 20 righe per trovare 'Giudizio'
        for row_idx, row in enumerate(sheet.iter_rows(min_row=1, max_row=20)):
            for cell in row:
                if isinstance(cell.value, str) and re.search(r'giudizio', str(cell.value).strip(), re.IGNORECASE):
                    return row_idx  # Restituisce l'indice di riga
        return None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione in '{sheet_name}': {e}")
        return None

def load_and_prepare_excel(file_data):
    """
    Legge un file Excel (in memoria) e prepara il corpus.

    Args:
        file_data (BytesIO): Il file Excel caricato in memoria.

    Returns:
        pd.DataFrame: Il corpus dei dati pronti per il fine-tuning.
    """
    corpus_list = []
    
    try:
        # Carica il file Excel in memoria
        file_data.seek(0)
        xls = pd.ExcelFile(file_data)
        sheet_names = xls.sheet_names

        for sheet_name in sheet_names:
            try:
                # Cerca l'indice della riga di intestazione
                header_row_index = find_header_row(BytesIO(file_data.getvalue()), sheet_name)
                
                if header_row_index is None:
                    print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Legge il foglio a partire dalla riga di intestazione trovata
                df_sheet = pd.read_excel(BytesIO(file_data.getvalue()), sheet_name=sheet_name, header=header_row_index)
                
                # Trova la colonna 'Giudizio' (case-insensitive)
                giudizio_col = find_giudizio_column(df_sheet)
                
                if giudizio_col is None:
                    print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Filtra le altre colonne che non sono 'Giudizio'
                other_cols = [col for col in df_sheet.columns if col != giudizio_col and pd.notna(col) and str(col).strip()]
                
                # Prepara la lista di dizionari per la creazione del dataset
                data_for_dataset = []
                for index, row in df_sheet.iterrows():
                    # Ignora le righe che non contengono alcun prompt valido
                    if all(pd.isna(row.get(col)) for col in other_cols):
                        continue

                    prompt_parts = []
                    for col in other_cols:
                        value = row.get(col)
                        if pd.notna(value) and str(value).strip():
                            prompt_parts.append(f"{col}: {str(value).strip()}")
                    
                    prompt_text = " ".join(prompt_parts)
                    target_text = str(row[giudizio_col]).strip() if pd.notna(row[giudizio_col]) else ""

                    # Aggiunge solo se c'è almeno un prompt valido
                    if prompt_text:
                        data_for_dataset.append({
                            'input_text': prompt_text,
                            'target_text': target_text
                        })
                
                if not data_for_dataset:
                    print(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                corpus_list.extend(data_for_dataset)
                
            except Exception as e:
                print(f"Errore nella lettura del foglio '{sheet_name}': {e}")
                
        if not corpus_list:
            print("Nessun dato valido trovato in tutti i fogli del file.")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        print(f"Errore nella lettura del file: {e}")
        return pd.DataFrame()
