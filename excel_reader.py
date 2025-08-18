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

def find_giudizio_column(df, file_path, sheet_name):
    """
    Trova la colonna 'Giudizio' nel DataFrame, cercando prima nella cella H3
    e poi in modo case-insensitive in tutte le intestazioni.
    
    Args:
        df (pd.DataFrame): Il DataFrame del foglio da analizzare.
        file_path (str): Il percorso del file Excel.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        str: Il nome della colonna 'Giudizio' o None se non trovata.
    """
    try:
        # Usa openpyxl per leggere la cella H3 (row=3, column=8)
        workbook = openpyxl.load_workbook(file_path, data_only=True)
        sheet = workbook[sheet_name]
        h3_value = sheet.cell(row=3, column=8).value
        
        # Cerca la parola 'giudizio' nella cella H3 e verifica che sia una colonna valida.
        if h3_value and 'giudizio' in str(h3_value).lower():
            # Cerca la colonna corrispondente al valore della cella H3
            # L'errore era qui: ora verifico che il valore di H3 sia una delle intestazioni del DataFrame.
            for col in df.columns:
                if str(col).lower() == str(h3_value).lower():
                    print(f"Colonna 'Giudizio' trovata in H3: '{col}'")
                    return col

    except Exception as e:
        # Se c'è un errore nella lettura della cella H3, procedi con la ricerca generica.
        print(f"Errore nella lettura della cella H3. Proseguo con la ricerca generica. Errore: {e}")
        pass

    # Logica di fallback: cerca in tutte le intestazioni in modo case-insensitive
    giudizio_keywords = ['giudizio', 'giudizi']
    for col in df.columns:
        for keyword in giudizio_keywords:
            if keyword in str(col).lower():
                print(f"Colonna 'Giudizio' trovata con ricerca generica: '{col}'")
                return col
    
    print("La colonna 'Giudizio' non è stata trovata.")
    return None

# ==============================================================================
# SEZIONE 3: FUNZIONE PRINCIPALE DI ELABORAZIONE
# ==============================================================================

def load_and_prepare_excel(file_path):
    """
    Carica un file Excel, ne elabora il contenuto e lo prepara per il fine-tuning.
    
    Args:
        file_path (str): Il percorso del file Excel da elaborare.
    
    Returns:
        pd.DataFrame: Un DataFrame con colonne 'input_text' e 'target_text'
                      pronte per il fine-tuning, o un DataFrame vuoto in caso di errore.
    """
    try:
        # Crea un oggetto ExcelFile per una gestione più efficiente.
        xl = pd.ExcelFile(file_path)
        
        corpus_list = []
        
        # Itera su tutti i fogli del file Excel.
        for sheet_name in xl.sheet_names:
            # Ignora i fogli non rilevanti.
            if any(kw in sheet_name.lower() for kw in ['prototipo', 'andamento', 'medie', 'copertina']):
                print(f"Foglio '{sheet_name}' ignorato.")
                continue

            # Legge il foglio in un DataFrame.
            df_sheet = pd.read_excel(xl, sheet_name=sheet_name)
            
            # Pulisce il DataFrame: rimuove righe e colonne completamente vuote.
            df_sheet.dropna(how='all', axis=0, inplace=True)
            df_sheet.dropna(how='all', axis=1, inplace=True)

            # Trova la riga di intestazione (solitamente la prima riga con dati significativi).
            header_row_index = -1
            for i, row in df_sheet.iterrows():
                if any(pd.notna(row)):
                    header_row_index = i
                    break
            
            if header_row_index == -1:
                print(f"Attenzione: Foglio '{sheet_name}' non contiene una riga di intestazione valida. Saltato.")
                continue

            # Imposta la riga di intestazione trovata e rilegge il file.
            df_sheet = pd.read_excel(xl, sheet_name=sheet_name, header=header_row_index)
            
            # Rimuove le righe vuote dopo il ricalcolo dell'intestazione.
            df_sheet.dropna(how='all', inplace=True)

            # Trova la colonna 'Giudizio'
            giudizio_col = find_giudizio_column(df_sheet, file_path, sheet_name)
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
        print(f"Errore nella lettura del file: {e}")
        return pd.DataFrame()
