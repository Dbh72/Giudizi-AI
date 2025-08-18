# excel_reader.py

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import openpyxl

# ==============================================================================
# SEZIONE 2: LOGICA DI GESTIONE DEI DATI
# ==============================================================================

def detect_header_row(df_sheet):
    """
    Individua la riga di intestazione (header) basandosi sull'assenza di valori numerici.
    
    Questa funzione analizza le prime 10 righe di un DataFrame e restituisce l'indice
    della prima riga che non contiene solo valori numerici.
    Questo aiuta a gestire i file Excel che hanno metadati o righe vuote prima dell'intestazione.
    """
    for i in range(min(10, len(df_sheet))):
        if not df_sheet.iloc[i].apply(lambda x: isinstance(x, (int, float))).all():
            return i
    return 0

def load_and_prepare_excel(file_path):
    """
    Carica un file Excel e prepara i dati per l'addestramento.
    
    Questa funzione legge tutti i fogli di lavoro da un file Excel, identifica
    l'intestazione e restituisce un dizionario con i DataFrame per ogni foglio.
    Viene utilizzata la stessa logica di lettura presente nel file 33 Funziona.txt.
    
    Args:
        file_path (str): Il percorso del file Excel.
        
    Returns:
        dict: Un dizionario dove le chiavi sono i nomi dei fogli e i valori sono i 
              rispettivi DataFrame. Ritorna None in caso di errore.
    """
    try:
        # Usiamo pd.ExcelFile per leggere il file e ottenere i nomi di tutti i fogli
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        extracted_dfs = {}
        for sheet_name in sheet_names:
            # Per ogni foglio, lo leggiamo come un DataFrame
            df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            # Troviamo la riga di intestazione
            header_row_index = detect_header_row(df_sheet)
            
            # Assegniamo l'intestazione e rimuoviamo le righe sopra di essa
            df_sheet.columns = df_sheet.iloc[header_row_index]
            df_sheet = df_sheet[header_row_index + 1:].reset_index(drop=True)
            
            extracted_dfs[sheet_name] = df_sheet
            
        return extracted_dfs
        
    except Exception as e:
        print(f"Errore durante l'elaborazione del file Excel: {e}")
        return None
