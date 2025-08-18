# excel_reader.py

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# pandas: per la manipolazione efficiente dei dati in formato DataFrame.
# openpyxl: per la gestione dei file Excel (.xlsx).
# os: per operazioni sul sistema operativo, come la gestione dei percorsi.
import pandas as pd
import openpyxl
import os

# ==============================================================================
# SEZIONE 2: LOGICHE DI LETTURA E PREPARAZIONE DATI
# ==============================================================================

def find_giudizio_column(df):
    """
    Trova la colonna 'Giudizio' (case-insensitive) nel DataFrame.
    """
    for col in df.columns:
        if isinstance(col, str) and "giudizio" in col.lower():
            return col
    return None

def find_pos_column(df):
    """
    Trova la colonna 'pos' (case-insensitive) nel DataFrame.
    """
    for col in df.columns:
        if isinstance(col, str) and col.lower().strip() == "pos":
            return col
    return None

def chunk_text_with_overlap(text, chunk_size=450, overlap=50):
    """
    Suddivide un testo in chunk di dimensione specificata con una sovrapposizione.
    Questo aiuta a non perdere il contesto nei modelli di linguaggio.
    
    Args:
        text (str): Il testo da suddividere.
        chunk_size (int): La dimensione massima di ogni chunk in numero di parole.
        overlap (int): Il numero di parole di sovrapposizione tra i chunk.
        
    Returns:
        list: Una lista di stringhe, dove ogni stringa è un chunk del testo originale.
    """
    words = text.split()
    chunks = []
    # Genera i chunk scorrendo le parole
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    # Gestisce il caso in cui l'ultimo chunk è molto piccolo e senza sovrapposizione,
    # unendolo al precedente per evitare chunk troppo corti
    if len(chunks) > 1 and len(words) % (chunk_size - overlap) < overlap:
        last_chunk = chunks.pop()
        if chunks: # Assicurati che ci sia almeno un elemento
            chunks[-1] += " " + last_chunk
        else: # Se era l'unico chunk, lo aggiungi di nuovo
            chunks.append(last_chunk)
            
    return chunks

def load_and_prepare_excel(file_path):
    """
    Carica un file Excel e prepara i dati in un DataFrame pulito.
    Questa funzione implementa le logiche di pulizia e troncamento avanzate
    come discusso nel riepilogo del progetto.
    """
    try:
        # Legge tutti i fogli del file Excel
        xls = pd.ExcelFile(file_path)
        all_sheets_data = []
        
        # Elenco dei fogli da ignorare, come da istruzioni
        sheets_to_ignore = ["prototipo", "medie"]

        for sheet_name in xls.sheet_names:
            if sheet_name.lower() in sheets_to_ignore:
                print(f"Foglio '{sheet_name}' ignorato.")
                continue

            df_sheet = pd.read_excel(xls, sheet_name=sheet_name, header=None)
            
            # --- Identificazione della riga di intestazione ---
            # Cerca la riga che contiene la colonna 'pos' per identificare l'intestazione
            header_row_index = -1
            for i, row in df_sheet.iterrows():
                # Cerchiamo la colonna con il nome 'pos' (case-insensitive)
                pos_col_found = None
                for col_idx, value in enumerate(row):
                    if isinstance(value, str) and value.lower().strip() == "pos":
                        pos_col_found = df_sheet.columns[col_idx]
                        break
                
                # Se troviamo 'pos' in una riga, la impostiamo come nuova intestazione
                if pos_col_found:
                    header_row_index = i
                    break
            
            if header_row_index != -1:
                df_sheet.columns = df_sheet.iloc[header_row_index]
                df_sheet = df_sheet.iloc[header_row_index + 1:].reset_index(drop=True)
            else:
                # Se non troviamo una riga di intestazione valida, saltiamo il foglio
                print(f"Attenzione: Foglio '{sheet_name}' non contiene una riga di intestazione valida. Saltato.")
                continue
            
            # --- Tronca le righe quando la sequenza numerica in "pos" si interrompe ---
            pos_col = find_pos_column(df_sheet)
            if pos_col:
                last_valid_row = -1
                for i, val in enumerate(df_sheet[pos_col]):
                    # Se il valore è un numero intero o una stringa che rappresenta un numero
                    if pd.notna(val) and (isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit())):
                        last_valid_row = i
                    else:
                        break # Interrompe la scansione al primo valore non numerico
                
                if last_valid_row != -1:
                    df_sheet = df_sheet.iloc[:last_valid_row + 1].copy()

            # --- Tronca le colonne dopo due colonne vuote consecutive ---
            empty_cols_count = 0
            cols_to_keep = []
            for col in reversed(df_sheet.columns):
                # Controlla se l'intera colonna è vuota (solo NaN)
                if df_sheet[col].isnull().all():
                    empty_cols_count += 1
                else:
                    empty_cols_count = 0
                
                if empty_cols_count < 2:
                    cols_to_keep.append(col)
                else:
                    break # Interrompe al raggiungimento di 2 colonne vuote consecutive
            
            # Mantiene solo le colonne valide
            df_sheet = df_sheet[list(reversed(cols_to_keep))]
            
            # Rimuove le righe completamente vuote rimanenti
            df_sheet.dropna(how='all', inplace=True)
            
            # Trova la colonna 'Giudizio'
            giudizio_col = find_giudizio_column(df_sheet)
            if not giudizio_col:
                print(f"Attenzione: Foglio '{sheet_name}' non contiene la colonna 'Giudizio'. Saltato.")
                continue
            
            # Identifica le altre colonne che devono essere usate per il prompt
            # Escludiamo la colonna 'Giudizio' e la colonna 'pos'
            other_cols = [col for col in df_sheet.columns if col != giudizio_col and col != 'pos' and pd.notna(col)]
            
            # Prepara la lista di dizionari per il dataset
            data_for_dataset = []
            for index, row in df_sheet.iterrows():
                prompt_parts = []
                for col in other_cols:
                    value = row.get(col)
                    # Controlliamo che il valore non sia vuoto
                    if pd.notna(value) and str(value).strip():
                        prompt_parts.append(f"{col}: {str(value).strip()}")
                
                prompt_text = " ".join(prompt_parts)
                target_text = str(row[giudizio_col]).strip() if pd.notna(row[giudizio_col]) else ""

                if prompt_text and target_text:
                    # Suddivide il testo del prompt in chunk se necessario, per evitare l'overflow dei token
                    if len(prompt_text.split()) > 450: # Usiamo un conteggio approssimativo per i token
                        chunks = chunk_text_with_overlap(prompt_text, chunk_size=400, overlap=50)
                        for chunk in chunks:
                            data_for_dataset.append({
                                'input_text': chunk,
                                'target_text': target_text # Il target rimane lo stesso per tutti i chunk
                            })
                    else:
                        data_for_dataset.append({
                            'input_text': prompt_text,
                            'target_text': target_text
                        })
            
            if data_for_dataset:
                all_sheets_data.append(pd.DataFrame(data_for_dataset))
        
        if not all_sheets_data:
            return pd.DataFrame()
        
        # Unisce i dati di tutti i fogli in un unico DataFrame
        return pd.concat(all_sheets_data, ignore_index=True)
        
    except Exception as e:
        print(f"Errore nella lettura del file: {e}")
        return pd.DataFrame()
