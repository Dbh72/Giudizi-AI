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
        int: L'indice della riga dell'intestazione (basato su 0) o None se non trovata.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 10 righe per trovare l'intestazione
        for i, row in enumerate(sheet.iter_rows(max_row=10, values_only=True)):
            if row is None:
                continue
            
            # Cerca una colonna che contenga 'giudizio' (case-insensitive)
            for cell_value in row:
                if isinstance(cell_value, str) and re.search(r'giudizio', cell_value, re.IGNORECASE):
                    # Trovato! Restituisci l'indice di riga (basato su 0)
                    return i
        
        return None
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione nel foglio '{sheet_name}': {e}")
        return None
    finally:
        # Riporta il cursore all'inizio del file per la successiva lettura di pandas
        file_path.seek(0)

# ==============================================================================
# SEZIONE 2: FUNZIONE PRINCIPALE DI CARICAMENTO E PREPARAZIONE
# ==============================================================================

def load_and_prepare_excel(file_path, progress_container=None):
    """
    Carica i dati da un file Excel e li prepara per il fine-tuning.

    Args:
        file_path (BytesIO): Il file Excel caricato in memoria.
        progress_container (list): Una lista per i messaggi di stato.

    Returns:
        pd.DataFrame: Un DataFrame combinato con le colonne 'input_text' e 'target_text'.
    """
    corpus_list = []

    try:
        # Carica il workbook per ottenere i nomi dei fogli
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet_names = workbook.sheetnames

        for sheet_name in sheet_names:
            try:
                # Trova la riga dell'intestazione
                header_row_index = find_header_row(file_path, sheet_name)
                if header_row_index is None:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Carica il foglio di lavoro nel DataFrame, ignorando le righe prima dell'intestazione
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
                
                # Trova la colonna 'Giudizio'
                giudizio_col = find_giudizio_column(df)
                if not giudizio_col:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet_name}'. Saltato.")
                    continue

                # Identifica le altre colonne che non sono 'Giudizio'
                other_cols = [col for col in df.columns if col != giudizio_col]
                if not other_cols:
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: Nessuna colonna di input trovata nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                data_for_dataset = []
                for _, row in df.iterrows():
                    # Salta se la colonna 'Giudizio' è vuota o contiene solo spazi bianchi
                    if pd.isna(row[giudizio_col]) or str(row[giudizio_col]).strip() == "":
                        continue

                    prompt_parts = []
                    for col in other_cols:
                        value = row.get(col)
                        # Salta le celle vuote o con solo spazi bianchi
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
                    if progress_container is not None:
                        progress_container.append(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet_name}'. Saltato.")
                    continue
                
                corpus_list.extend(data_for_dataset)
                
            except Exception as e:
                if progress_container is not None:
                    progress_container.append(f"Errore nella lettura del foglio '{sheet_name}': {e}")
                    progress_container.append(traceback.format_exc())
                
        if not corpus_list:
            if progress_container is not None:
                progress_container.append("Nessun dato valido trovato in tutti i fogli del file.")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        if progress_container is not None:
            progress_container.append(f"Errore nella lettura del file: {e}")
            progress_container.append(traceback.format_exc())
        return pd.DataFrame()
