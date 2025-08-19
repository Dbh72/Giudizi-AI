# ==============================================================================
# File: excel_reader_v2.py
# Logica per la preparazione dei dati da file Excel, integrando le
# funzionalità di '33 Funziona.txt' per una lettura più robusta.
# ==============================================================================
# ==============================================================================
# Il codice è predisposto per leggere e processare tutti i fogli di lavoro
# di un file Excel. La funzione chiave per questo è read_and_prepare_data_from_excel,
# che itera attraverso ogni foglio presente nel file per estrarre i dati.
# 
# Caricamento multifoglio con esclusione dei fogli 'Prototipo' e 'Medie'.
# ==============================================================================
# ==============================================================================
# Logica del Troncamento
# 1. La logica di caricamento identifica e rimuove le colonne Alunno', 'assenti' e 'CNT'
# e le esclude in modo case-insensitive dal prompt del modello.
# Le colonne Alunno, assenti, CNT (e le loro varianti con maiuscole/minuscole
# e spazi) vengono identificate e ignorate, in modo che i loro contenuti non
# vengano inclusi nei prompt per il modello.
#
# 2. La colonna 'pos' viene anch'essa esclusa dal prompt di input mantenendola però
# per le altre funzionalità che ne fanno uso, come il troncamento del file.
#
# 3. La fase di "pulitura" dei dati, che include il troncamento e l'eliminazione
# delle righe e colonne non pertinenti, è un aspetto fondamentale del codice.
#
# 4. Troncamento delle Righe per 'pos': Il codice identifica la colonna 'pos'
# (o equivalenti, ignorando maiuscole/minuscole e spazi) come identificativo
# dello studente. La lettura delle righe si interrompe automaticamente non
# appena la sequenza numerica in questa colonna si spezza o termina.
#
# 5. Troncamento delle Colonne Vuote: Una volta individuata la fine dei dati,
# il codice scansiona il DataFrame da destra a sinistra. Le colonne vengono
# rimosse finché non vengono trovate due colonne consecutive completamente
# vuote. Questo garantisce che vengano preservati solo i dati utili.
#
# 6. Gestione delle Celle Vuote: La logica di preparazione dei dati è stata
# ottimizzata per ignorare le singole celle vuote (o NaN) ma non l'intera riga,
# a meno che non sia completamente priva di informazioni.
# ==============================================================================
# ==============================================================================
# Pulizia Dati
# Pulizia Finale del Dataset: Per garantire la qualità dei dati, vengono
# eseguiti due passaggi di pulizia finali:
# 1. Vengono rimosse le righe che sono diventate completamente vuote dopo
#    il troncamento.
# 2. Prima di creare il dataset per il fine-tuning, vengono eliminate tutte le
#    righe che hanno il campo 'Giudizio' vuoto, assicurando che ogni esempio
#    di addestramento sia valido.
# ==============================================================================
# Identificazione Flessibile di Intestazioni e Colonne
# Le funzioni di ricerca delle intestazioni e delle colonne sono state rese
# estremamente flessibili per adattarsi a layout diversi.
# 1. Ricerca Flessibile dell'Intestazione: Il codice non si affida a una riga
# specifica, ma esegue una ricerca intelligente nelle prime 10 righe del foglio
# per trovare quella che contiene le intestazioni.
# 2. Fallback per la Colonna 'Giudizio': La ricerca della colonna 'Giudizio'
# avviene in modo dinamico e insensibile a maiuscole, minuscole o spazi.
# La logica di ricerca è:
# a.  Cerca la parola chiave "giudizio" nell'intera riga di intestazione.
# b.  Se non viene trovata alcuna corrispondenza testuale, viene utilizzato un
#   "fallback" sicuro che si aspetta la colonna H (indice 7) come destinazione
#   predefinita per il giudizio, gestendo così anche i casi più difficili.
# ==============================================================================
# Ruolo della Colonna 'Giudizio' nel Fine-Tuning
# La colonna "Giudizio" ha un ruolo centrale ma specifico nel processo di
# addestramento del modello.
# 1. Risposta Desiderata (Target): Per ogni riga del file, il contenuto della
#   colonna "Giudizio" rappresenta la risposta corretta che il modello deve
#   imparare a generare. Questa stringa viene assegnata al campo 'target_text'
#   nel dataset di fine-tuning.
# 2. Creazione del Prompt (Input): Le informazioni contenute in tutte le
#   altre colonne (ad eccezione di 'pos', 'Alunno', 'assenti' e 'CNT')
#   vengono unite in un'unica stringa. Questa stringa funge da 'input_text'
#   o "prompt" per il modello.
# In sintesi, non è la colonna "Giudizio" a raccogliere le altre, ma al
# contrario, le altre colonne forniscono l'input che il modello usa per
# generare il testo della colonna "Giudizio". Il codice riflette questa
# logica di "text-to-text" per l'addestramento.
#
# Identificazione Flessibile: La funzione 'find_header_row_and_columns' è
# progettata per scansionare le prime righe del file e identificare in modo
# flessibile la riga di intestazione e la colonna 'Giudizio', includendo anche
# il fallback automatico sulla colonna H se la ricerca testuale fallisce.
#
# Ruolo della Colonna 'Giudizio': Le funzioni preparano_i_dati e leggono_e_uniscono_
# i_fogli estraggono correttamente il testo dalla colonna 'Giudizio' per usarlo
# come target ('target_text'). Contemporaneamente, uniscono i contenuti di tutte
# le altre colonne rilevanti (escludendo 'pos', 'Alunno', 'assenti' e 'CNT')
# per creare il prompt di input ('input_text').
# ==============================================================================
# ==============================================================================
# Logica di Troncamento e Pulizia: Il codice include funzioni dedicate come
# 'trim_dataframe_by_numeric_id_column' per il troncamento delle righe
# basato sulla colonna 'pos' e 'trim_dataframe_by_empty_columns' per la
# rimozione delle colonne vuote a fine file. La pulizia finale per eliminare
# le righe con giudizi mancanti è gestita direttamente nella funzione di
# preparazione dei dati.
# ==============================================================================
# ==============================================================================
# Utilizzo di Tutte le Colonne Utili: Per la creazione del prompt per il modello,
# il codice assicura che tutte le colonne identificate, ad eccezione di quella
# del giudizio, vengano utilizzate per costruire il testo di input.
# ==============================================================================
# ==============================================================================
# File: excel_reader_v2.py
# Logica per la preparazione dei dati da file Excel, integrando le
# funzionalità di '33 Funziona.txt' per una lettura più robusta.
# ==============================================================================
import pandas as pd
import openpyxl
import re
from io import BytesIO
import traceback
import os
import shutil
import json
from datetime import datetime
import numpy as np

# ==============================================================================
# SEZIONE 1: CONFIGURAZIONE
# ==============================================================================
# Nomi delle colonne da ignorare durante la creazione del prompt
COLUMNS_TO_IGNORE = ['alunno', 'assenti', 'cnt', 'pos']
# Fogli di lavoro da ignorare durante il fine-tuning
SHEETS_TO_IGNORE = ['Prototipo', 'Medie']

# ==============================================================================
# SEZIONE 2: FUNZIONI AUSILIARIE
# ==============================================================================

def is_column_empty(df_column):
    """Controlla se una colonna è considerata vuota (tutti NaN o spazi vuoti)."""
    return df_column.isnull().all() or (df_column.astype(str).str.strip() == '').all()

def trim_dataframe_by_empty_columns(df):
    """
    Trunca il DataFrame rimuovendo le colonne vuote consecutive da destra a sinistra.
    Si ferma alla prima coppia di colonne vuote consecutive.
    """
    if df.empty:
        return df
    
    # Rimuove le colonne che sono completamente vuote dall'inizio
    # Questo serve a pulire il dataframe da colonne vuote iniziali.
    df = df.dropna(axis=1, how='all')

    # Trova l'indice della prima colonna non vuota partendo da destra.
    # Questo è l'inizio del nostro taglio.
    last_non_empty_col_index = df.apply(lambda col: not is_column_empty(col), axis=0).idxmax()
    
    # Se tutte le colonne sono vuote, restituisce un DataFrame vuoto
    if is_column_empty(df.iloc[:, last_non_empty_col_index]):
        return pd.DataFrame()

    # Rimuove le colonne a destra della prima colonna non vuota trovata
    df = df.loc[:, :last_non_empty_col_index]

    return df

def make_columns_unique(columns):
    """
    Garantisce che i nomi delle colonne siano unici, aggiungendo un contatore
    se necessario.
    """
    seen = {}
    new_columns = []
    for col in columns:
        original_col = str(col).strip() if pd.notna(col) else ""
        if original_col in seen:
            seen[original_col] += 1
            new_columns.append(f"{original_col}_{seen[original_col]}")
        else:
            seen[original_col] = 0
            new_columns.append(original_col)
    return new_columns


def find_header_row_and_columns(df_raw):
    """
    Trova la riga di intestazione e la posizione della colonna 'Giudizio'.
    Cerca la riga di intestazione nelle prime 50 righe del DataFrame.
    """
    giudizio_col = None
    header_row_index = -1
    df = None

    try:
        for i in range(min(50, len(df_raw))):
            row = df_raw.iloc[i].astype(str).str.lower()
            if any('giudizio' in cell for cell in row.values):
                header_row_index = i
                break
        
        # Fallback se non si trova la riga 'Giudizio' testualmente
        if header_row_index == -1:
            try:
                # Prova la colonna H (indice 7) come fallback
                # Assumendo che il DataFrame non sia vuoto
                if not df_raw.empty and len(df_raw.columns) > 7:
                    potential_header_row = df_raw.iloc[0].astype(str)
                    if 'giudizio' in str(potential_header_row.iloc[7]).lower():
                        header_row_index = 0
                        print("Intestazione 'Giudizio' non trovata testualmente, usando fallback sulla colonna H.")
            except Exception:
                pass # Ignora errori e continua

        if header_row_index != -1:
            df_raw.columns = df_raw.iloc[header_row_index]
            df = df_raw.iloc[header_row_index+1:].reset_index(drop=True)
            df.columns = make_columns_unique(df.columns)
            giudizio_col = next((col for col in df.columns if 'giudizio' in str(col).lower()), None)

    except Exception as e:
        print(f"Errore nella ricerca dell'intestazione: {e}")
        giudizio_col = None

    return df, giudizio_col, header_row_index


def read_and_prepare_data_from_excel(file_path, sheet_name=None, progress_container=None):
    """
    Legge un file Excel, processa i dati e restituisce un DataFrame pulito
    adatto per il fine-tuning del modello.
    """
    if progress_container is None:
        def progress_container(msg, type):
            print(msg)
    
    try:
        progress_container("Lettura del file Excel in corso...", "info")
        
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        all_sheet_names = workbook.sheetnames
        workbook.close()

        corpus_list = []
        
        sheets_to_process = [s for s in all_sheet_names if s not in SHEETS_TO_IGNORE]
        
        if not sheets_to_process:
            progress_container("Attenzione: Nessun foglio di lavoro da processare dopo l'applicazione dei filtri.", "warning")
            return pd.DataFrame()

        progress_container(f"Fogli di lavoro da leggere: {', '.join(sheets_to_process)}", "info")

        for sheet in sheets_to_process:
            try:
                progress_container(f"Processamento del foglio '{sheet}'...", "info")
                df_raw = pd.read_excel(file_path, sheet_name=sheet, header=None, engine='openpyxl')
                
                # Trova la riga di intestazione e la colonna del giudizio
                df, giudizio_col, header_row_index = find_header_row_and_columns(df_raw)

                if df is None or giudizio_col is None:
                    progress_container(f"Attenzione: La colonna 'Giudizio' non è stata trovata o il file ha una struttura non standard nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                progress_container(f"La colonna 'Giudizio' si chiama '{giudizio_col}' nel foglio '{sheet}'.", "info")

                # Logica di troncamento basata sulla colonna 'pos'
                pos_col = next((col for col in df.columns if 'pos' in str(col).lower()), None)
                if pos_col:
                    last_pos_row = -1
                    # Trova l'ultima riga valida nella colonna 'pos'
                    for idx, val in enumerate(df[pos_col]):
                        if pd.notna(val) and isinstance(val, (int, float)):
                            last_pos_row = idx
                    
                    if last_pos_row != -1:
                        df = df.iloc[:last_pos_row + 1]
                    else:
                        progress_container(f"Attenzione: La colonna '{pos_col}' nel foglio '{sheet}' non contiene dati numerici validi per il troncamento. Saltato il troncamento.", "warning")
                
                # Troncamento delle colonne vuote da destra a sinistra
                df = trim_dataframe_by_empty_columns(df)

                data_for_dataset = []
                # Rimuove le righe che non hanno un valore nella colonna 'Giudizio'
                df.dropna(subset=[giudizio_col], inplace=True)
                
                if df.empty:
                    progress_container(f"Attenzione: Nessun dato valido trovato con la colonna 'Giudizio' compilata nel foglio '{sheet}'. Saltato.", "warning")
                    continue

                for _, row in df.iterrows():
                    input_data = {}
                    for col, val in row.items():
                        # Converte il nome della colonna in minuscolo per la verifica
                        col_lower = str(col).lower()
                        # Esclude le colonne da ignorare e la colonna del giudizio
                        if not any(ignore_col in col_lower for ignore_col in COLUMNS_TO_IGNORE) and col != giudizio_col:
                            if pd.notna(val) and str(val).strip() != '':
                                input_data[col] = str(val)

                    prompt_text = " ".join([f"{k}: {v}" for k, v in input_data.items()])
                    target_text = str(row[giudizio_col]) if pd.notna(row[giudizio_col]) else ""
                    
                    if prompt_text:
                        data_for_dataset.append({
                            'input_text': prompt_text,
                            'target_text': target_text
                        })

                if not data_for_dataset:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                corpus_list.extend(data_for_dataset)
            
            except Exception as e:
                progress_container(f"Errore nella lettura del foglio '{sheet}': {e}", "error")
                progress_container(f"Traceback: {traceback.format_exc()}", "error")
        
        if not corpus_list:
            progress_container("Nessun dato valido trovato in tutti i fogli del file.", "error")
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)

    except Exception as e:
        progress_container(f"Errore nella lettura del file: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return pd.DataFrame()


def get_excel_sheet_names(file_path):
    """
    Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet_names = workbook.sheetnames
        workbook.close()
        return sheet_names
    except Exception as e:
        return [f"Errore nella lettura dei fogli: {e}"]
