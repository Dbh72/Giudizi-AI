# excel_reader.py - Logica di preparazione dei dati da file Excel

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import openpyxl
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
        int: L'indice della riga dell'intestazione (0-based) o None se non trovata.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet = workbook[sheet_name]
        
        # Scansiona le prime 10 righe per l'intestazione
        for row_idx, row in enumerate(sheet.iter_rows(min_row=1, max_row=10, values_only=True)):
            if 'giudizio' in [str(cell).strip().lower() for cell in row if cell is not None]:
                return row_idx
    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione: {e}")
    return None


def load_and_prepare_excel(file_path):
    """
    Legge un file Excel, identifica i fogli e la colonna 'Giudizio' e prepara
    un DataFrame per il fine-tuning.

    Args:
        file_path (str): Il percorso del file Excel.

    Returns:
        pd.DataFrame: Un DataFrame unificato con le colonne 'input_text' e 'target_text'.
    """
    corpus_list = []
    
    try:
        # Usa openpyxl per ottenere tutti i nomi dei fogli
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        all_sheet_names = workbook.sheetnames

        # Filtra i fogli che contengono dati validi
        relevant_sheets = [s for s in all_sheet_names if s not in ['Medie', 'Prototipo', 'copertina']]
        
        # Se non ci sono fogli validi, restituisci un DataFrame vuoto
        if not relevant_sheets:
            print("Nessun foglio valido trovato.")
            return pd.DataFrame()

        for sheet_name in relevant_sheets:
            print(f"Elaborazione del foglio '{sheet_name}'...")
            
            # Trova la riga dell'intestazione.
            header_row_index = find_header_row(file_path, sheet_name)
            
            if header_row_index is None:
                print(f"Attenzione: Impossibile trovare la riga dell'intestazione nel foglio '{sheet_name}'. Saltato.")
                continue

            # Legge il foglio di lavoro nel DataFrame, saltando le righe prima dell'intestazione.
            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row_index)
            
            # Pulisce le righe e le colonne vuote
            df_sheet.dropna(how='all', inplace=True)
            df_sheet.dropna(axis=1, how='all', inplace=True)

            # Trova la colonna 'Giudizio'
            giudizio_col = find_giudizio_column(df_sheet)
            
            if not giudizio_col:
                print(f"Attenzione: La colonna 'Giudizio' non è stata trovata nel foglio '{sheet_name}'. Saltato.")
                continue
                
            # Identifica le altre colonne che devono essere usate per il prompt
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
        return pd.DataFrame()

