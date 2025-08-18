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
        int: L'indice della riga dell'intestazione (0-based) o None se non trovata.
    """
    try:
        wb = openpyxl.load_workbook(file_path, read_only=True)
        ws = wb[sheet_name]

        for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=10)):
            for cell in row:
                if cell.value and isinstance(cell.value, str) and re.search(r'giudizio', cell.value, re.IGNORECASE):
                    wb.close()
                    return row_idx
        wb.close()
    except Exception:
        # Ignora gli errori di lettura e continua a cercare
        pass
    return None

def load_and_prepare_excel(file_data, file_name):
    """
    Legge i dati da uno o più fogli di un file Excel e li prepara
    per il fine-tuning del modello.

    Crea un corpus combinando 'input_text' e 'target_text' da ogni riga
    di tutti i fogli di lavoro.

    Args:
        file_data (BytesIO): Il file Excel caricato in memoria.
        file_name (str): Il nome del file Excel.

    Returns:
        pd.DataFrame: Un DataFrame combinato con le colonne 'input_text' e 'target_text'.
                      Restituisce un DataFrame vuoto in caso di errore o se non trova dati validi.
    """
    corpus_list = []
    
    try:
        xls = pd.ExcelFile(file_data)
        sheet_names = xls.sheet_names
        
        for sheet_name in sheet_names:
            try:
                header_row = find_header_row(BytesIO(file_data.getvalue()), sheet_name)
                
                if header_row is None:
                    print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}' del file '{file_name}'. Saltato.")
                    continue
                
                df_sheet = pd.read_excel(file_data, sheet_name=sheet_name, header=header_row)
                
                giudizio_col = find_giudizio_column(df_sheet)
                
                if giudizio_col is None:
                    print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}' del file '{file_name}'. Saltato.")
                    continue
                
                other_cols = [col for col in df_sheet.columns if col != giudizio_col]
                df_sheet.dropna(how='all', subset=other_cols, inplace=True)
                df_sheet.reset_index(drop=True, inplace=True)
                
                data_for_dataset = []
                for index, row in df_sheet.iterrows():
                    # Salta la riga se la colonna 'Giudizio' è vuota e non ci sono altri dati
                    if pd.isna(row[giudizio_col]) and all(pd.isna(row[col]) for col in other_cols):
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
                    print(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}' del file '{file_name}'. Saltato.")
                    continue
                
                corpus_list.extend(data_for_dataset)
                
            except Exception as e:
                print(f"Errore nella lettura del foglio '{sheet_name}' del file '{file_name}': {e}")
                
        if not corpus_list:
            print(f"Nessun dato valido trovato in tutti i foghi del file '{file_name}'.")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        print(f"Errore nella lettura del file '{file_name}': {e}")
        return pd.DataFrame()
