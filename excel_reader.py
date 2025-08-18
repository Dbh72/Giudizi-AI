# ==============================================================================
# File: excel_reader.py
# Logica per la preparazione dei dati da file Excel
# ==============================================================================
import pandas as pd
import openpyxl
import re
from io import BytesIO
import traceback

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
        int: L'indice (base 0) della riga dell'intestazione o None se non trovata.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 100 righe per trovare l'intestazione
        for i, row in enumerate(sheet.iter_rows()):
            header_row = [cell.value for cell in row]
            for col_name in header_row:
                if isinstance(col_name, str) and re.search(r'giudizio', col_name, re.IGNORECASE):
                    return i
        return None
    except Exception as e:
        print(f"Errore nella ricerca della riga di intestazione: {e}")
        return None
        
def prepare_training_data(file_path):
    """
    Prepara un DataFrame di addestramento da un file Excel, leggendo tutti i fogli
    e cercando le colonne 'input' e 'target' (o 'giudizio').

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.

    Returns:
        pd.DataFrame: Un DataFrame con colonne 'input_text' e 'target_text' o un DataFrame vuoto.
    """
    try:
        file_path.seek(0)
        excel_file = pd.ExcelFile(file_path)
        all_data = []
        
        for sheet_name in excel_file.sheet_names:
            try:
                # Carica il DataFrame completo per trovare la riga dell'intestazione
                header_row_index = find_header_row(file_path, sheet_name)
                if header_row_index is None:
                    print(f"Attenzione: Riga di intestazione non trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
                
                # Trova le colonne "input" e "target"
                input_col = None
                giudizio_col = None
                for col in df.columns:
                    if isinstance(col, str):
                        if re.search(r'(input|descrizione|commento|testo)', col, re.IGNORECASE):
                            input_col = col
                        if re.search(r'(giudizio|giudizi|output)', col, re.IGNORECASE):
                            giudizio_col = col
                
                if input_col and giudizio_col:
                    temp_df = df.rename(columns={input_col: 'input_text', giudizio_col: 'target_text'})
                    temp_df = temp_df[['input_text', 'target_text']].dropna()
                    
                    if not temp_df.empty:
                        all_data.append(temp_df)
                else:
                    print(f"Attenzione: Colonne 'input' o 'giudizio' non trovate nel foglio '{sheet_name}'. Saltato.")
                    
            except Exception as e:
                print(f"Errore nella lettura del foglio '{sheet_name}': {e}")
                
        if not all_data:
            print("Nessun dato valido trovato in tutti i fogli del file per l'addestramento.")
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

    except Exception as e:
        print(f"Errore nella lettura del file per l'addestramento: {e}")
        print(traceback.format_exc())
        return pd.DataFrame()


def prepare_dataframe_to_complete(file_path, sheet_name):
    """
    Prepara un DataFrame da completare, individuando la colonna 'Giudizio'.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        tuple: (DataFrame da completare, nome della colonna 'Giudizio') o (None, None).
    """
    try:
        file_path.seek(0)
        
        # Cerca la riga di intestazione
        header_row_index = find_header_row(file_path, sheet_name)
        if header_row_index is None:
            return None, None

        # Legge il DataFrame con l'intestazione corretta
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)

        # Trova la colonna del giudizio
        giudizio_col = find_giudizio_column(df)
        if giudizio_col is None:
            return None, None
            
        # Trova la colonna "input" per il prompt. Cerca tra le possibili
        # intestazioni "descrizione", "commento", "input", "testo".
        input_col = None
        for col in df.columns:
            if isinstance(col, str) and re.search(r'(input|descrizione|commento|testo)', col, re.IGNORECASE):
                input_col = col
                break
        
        if input_col is None:
            print("Colonna 'input' non trovata. La prima colonna verrà usata come prompt.")
            input_col = df.columns[0]
            
        # Ritorna il DataFrame e il nome della colonna del giudizio
        return df, giudizio_col
        
    except Exception as e:
        print(f"Errore nella preparazione del DataFrame: {e}")
        print(traceback.format_exc())
        return None, None
