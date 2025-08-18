# excel_reader.py - Logica di preparazione dei dati da file Excel

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import openpyxl
import os
import re

# ==============================================================================
# SEZIONE 2: FUNZIONI AUSILIARIE
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
        file_path (str): Il percorso del file Excel.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        int: L'indice di riga dell'intestazione o None se non trovata.
    """
    try:
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 10 righe per trovare 'Giudizio'.
        for row_idx in range(1, 11):
            for col_idx in range(1, sheet.max_column + 1):
                cell_value = str(sheet.cell(row=row_idx, column=col_idx).value).strip()
                if re.search(r'giudizio', cell_value, re.IGNORECASE):
                    return row_idx - 1 # Ritorniamo l'indice di riga (0-based)
        return None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione del file: {e}")
        return None


def load_and_prepare_excel(file_path):
    """
    Carica un file Excel, identifica le colonne e i dati rilevanti,
    e restituisce un DataFrame di pandas per il fine-tuning.

    Args:
        file_path (str): Il percorso del file Excel da elaborare.

    Returns:
        pd.DataFrame: Un DataFrame contenente le colonne 'input_text' e
                      'target_text' per il fine-tuning, o un DataFrame vuoto
                      in caso di errore.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        all_sheets = workbook.sheetnames
        corpus_list = []
        
        for sheet_name in all_sheets:
            # Ignoriamo i fogli non pertinenti (es. 'copertina').
            if "copertina" in sheet_name.lower():
                print(f"Attenzione: Foglio '{sheet_name}' ignorato.")
                continue
            
            # Troviamo la riga di intestazione
            header_row_index = find_header_row(file_path, sheet_name)
            if header_row_index is None:
                print(f"Attenzione: Riga di intestazione con 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                continue

            # Carichiamo il foglio come DataFrame, usando l'intestazione trovata
            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)

            # Troviamo la colonna 'Giudizio' nel DataFrame del foglio
            giudizio_col = find_giudizio_column(df_sheet)
            if giudizio_col is None:
                print(f"Attenzione: Colonna 'Giudizio' non trovata nel foglio '{sheet_name}'. Saltato.")
                continue
            
            # Filtriamo le righe che contengono dati non validi nella colonna 'Giudizio'
            df_sheet = df_sheet.dropna(subset=[giudizio_col])
            
            # Togliamo le colonne con solo valori NaN o una sola riga
            df_sheet = df_sheet.dropna(axis=1, how='all')
            df_sheet = df_sheet.dropna(axis=0, how='all')
            
            # Troviamo tutte le altre colonne
            other_cols = [col for col in df_sheet.columns if col != giudizio_col]
            
            # Prepara la lista di dizionari per la creazione del dataset
            data_for_dataset = []
            for index, row in df_sheet.iterrows():
                prompt_parts = []
                for col in other_cols:
                    value = row.get(col)
                    if pd.notna(value) and str(value).strip():
                        prompt_parts.append(f"{col}: {str(value).strip()}")
                
                prompt_text = " ".join(prompt_parts)
                target_text = str(row[giudizio_col]).strip() if pd.notna(row[giudizio_col]) else ""

                if prompt_text and target_text:
                    data_for_dataset.append({
                        'input_text': prompt_text,
                        'target_text': target_text
                    })
            
            if not data_for_dataset:
                print(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                continue
                
            corpus_list.extend(data_for_dataset)
        
        # Se non c'è nessun dato utile in tutti i fogli.
        if not corpus_list:
            print("Nessun dato valido trovato in tutti i fogli del file.")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)
        
    except Exception as e:
        print(f"Errore nella lettura del file '{os.path.basename(file_path)}': {e}")
        traceback_str = traceback.format_exc()
        print(traceback_str)
        return pd.DataFrame()

