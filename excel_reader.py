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
        int: L'indice della riga di intestazione (basato su 0) o None se non trovata.
    """
    try:
        # Apriamo il file con openpyxl per scansionare le righe
        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 10 righe
        for i, row in enumerate(sheet.iter_rows(min_row=1, max_row=10)):
            for cell in row:
                if isinstance(cell.value, str) and re.search(r'giudizio', cell.value, re.IGNORECASE):
                    return i # Restituisce l'indice della riga (basato su 0)
        return None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione per il foglio '{sheet_name}': {e}")
        return None

def load_and_prepare_excel(uploaded_files):
    """
    Carica i dati da una lista di file Excel caricati, identifica la colonna di giudizio
    e crea un corpus di addestramento. Gestisce più fogli di lavoro e file.

    Args:
        uploaded_files (list): Una lista di oggetti file caricati da Streamlit.

    Returns:
        pd.DataFrame: Un DataFrame unificato per il fine-tuning o un DataFrame vuoto in caso di errore.
    """
    corpus_list = []
    
    for uploaded_file in uploaded_files:
        try:
            file_path = BytesIO(uploaded_file.getvalue())
            file_name = uploaded_file.name

            xls = pd.ExcelFile(file_path)
            sheet_names = xls.sheet_names
            
            for sheet_name in sheet_names:
                print(f"Lavorazione del file '{file_name}', foglio '{sheet_name}'...")
                
                # Trova la riga di intestazione
                header_row_index = find_header_row(file_path, sheet_name)
                if header_row_index is None:
                    print(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                # Leggi il DataFrame a partire dalla riga dell'intestazione
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
                
                # Identifica la colonna 'Giudizio'
                giudizio_col = find_giudizio_column(df)
                if not giudizio_col:
                    print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Rimuovi le colonne non necessarie e le righe completamente vuote
                df = df.dropna(how='all')
                df = df.drop(columns=[col for col in df.columns if 'unnamed' in str(col).lower()], errors='ignore')

                other_cols = [col for col in df.columns if col != giudizio_col]
                data_for_dataset = []

                # Itera sulle righe per creare i dati del dataset
                for index, row in df.iterrows():
                    # Salta le righe dove la colonna 'Giudizio' è vuota
                    if pd.isna(row.get(giudizio_col, None)) or not str(row[giudizio_col]).strip():
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
            print(f"Errore nella lettura del file '{uploaded_file.name}': {e}\n{traceback.format_exc()}")
            
    if not corpus_list:
        print("Nessun dato valido trovato in tutti i file.")
        return pd.DataFrame()
        
    return pd.DataFrame(corpus_list)
