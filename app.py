# ==============================================================================
# File: app.py
# L'interfaccia utente principale per l'applicazione di generazione di giudizi.
# Utilizza Streamlit per creare un'interfaccia web interattiva.
# Questo file gestisce l'intero flusso di lavoro:
# 1. Caricamento dei file di addestramento e/o del modello fine-tunato.
# 2. Addestramento (fine-tuning) del modello.
# 3. Caricamento del file Excel da completare.
# 4. Generazione dei giudizi per il file caricato.
# 5. Download del file Excel completato.
# ==============================================================================

# SEZIONE 0: LIBRERIE NECESSARIE E CONFIGURAZIONE
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import shutil
import warnings
import traceback
from datetime import datetime
from io import BytesIO
import json
from datasets import Dataset, DatasetDict
import re
import time
import sys

# Importa i moduli personalizzati
# Ho modificato l'import per usare il nuovo file excel_reader_v2
import excel_reader_v2 as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, CORPUS_FILE, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def log_and_display_message(message, type="info"):
    """
    Funzione per loggare e visualizzare un messaggio di stato nell'interfaccia utente.
    """
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    
    st.session_state.status_messages.append({'message': message, 'type': type})
    st.rerun()

def get_last_completed_row(df, column_name):
    """
    Trova l'indice dell'ultima riga completata in una colonna, ignorando le
    righe con valori NaN o stringhe vuote.
    """
    # Inverti il DataFrame per cercare dall'ultimo valore
    df_reversed = df.iloc[::-1]
    
    # Trova il primo indice di riga che non è NaN e non è una stringa vuota
    last_completed_row = df_reversed[df_reversed[column_name].astype(str).str.strip() != ''].index
    
    # Se un indice è stato trovato, restituisci l'indice+1 (la riga successiva)
    if not last_completed_row.empty:
        return last_completed_row[0] + 1
    
    # Se non è stato trovato nessun valore, ritorna 0 per iniziare dall'inizio
    return 0

# ==============================================================================
# SEZIONE 2: LAYOUT E LOGICA DELL'INTERFACCIA UTENTE
# ==============================================================================

# Configura il layout di Streamlit
st.set_page_config(layout="wide")
st.title("Sistema di Generazione Giudizi con Fine-Tuning")

# Inizializzazione delle session_state
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'training_df' not in st.session_state:
    st.session_state.training_df = pd.DataFrame()
if 'excel_df' not in st.session_state:
    st.session_state.excel_df = pd.DataFrame()
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'file_uploaded_for_training' not in st.session_state:
    st.session_state.file_uploaded_for_training = False
if 'file_uploaded_for_generation' not in st.session_state:
    st.session_state.file_uploaded_for_generation = False
if 'sheets_for_generation' not in st.session_state:
    st.session_state.sheets_for_generation = []
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'model_state' not in st.session_state:
    st.session_state.model_state = 'NO_MODEL' # 'NO_MODEL', 'READY', 'TRAINING'
if 'processed_rows' not in st.session_state:
    st.session_state.processed_rows = 0
if 'total_rows' not in st.session_state:
    st.session_state.total_rows = 0

# ==============================================================================
# SEZIONE 2.1: CARICAMENTO O ADDESTRAMENTO DEL MODELLO
# ==============================================================================

st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel con una colonna 'Giudizio' già compilata per addestrare o aggiornare il modello. Il processo creerà un corpus di addestramento.")
st.markdown("---")

fine_tune_file = st.file_uploader(
    "Carica file di addestramento", 
    type=['xlsx', 'xls', 'xlsm'], 
    key='fine_tune_file_uploader'
)

col1, col2 = st.columns(2)

with col1:
    train_button = st.button("Avvia Addestramento", disabled=not fine_tune_file, key='train_button')

with col2:
    if st.session_state.model_state == 'NO_MODEL':
        load_model_button = st.button("Carica Modello Esistente", key='load_model_button')
    else:
        # Se il modello è già stato caricato, mostra un messaggio invece del bottone
        st.info("Modello caricato.")

if train_button and fine_tune_file:
    log_and_display_message(f"File '{fine_tune_file.name}' caricato per l'addestramento.", "info")
    st.session_state.file_uploaded_for_training = True
    
    # --- RIGA MODIFICATA ---
    # Il nome della funzione è stato cambiato in read_and_prepare_data_from_excel
    st.session_state.training_df = er.read_and_prepare_data_from_excel(BytesIO(fine_tune_file.getvalue()), log_and_display_message)
    # -----------------------

    if not st.session_state.training_df.empty:
        log_and_display_message(f"Dati di addestramento letti con successo. Trovate {len(st.session_state.training_df)} righe valide.", "success")
        st.session_state.model_state = 'TRAINING'
        st.session_state.training_df = cb.build_or_update_corpus(st.session_state.training_df, log_and_display_message)
        
        # Avvia il fine-tuning solo se il corpus è valido
        if not st.session_state.training_df.empty:
            st.session_state.model, st.session_state.tokenizer = mt.fine_tune_model(st.session_state.training_df, log_and_display_message)
            if st.session_state.model and st.session_state.tokenizer:
                st.session_state.model_state = 'READY'
    else:
        log_and_display_message("Nessun dato valido trovato nel file di addestramento. Impossibile avviare l'addestramento.", "error")
        st.session_state.model_state = 'NO_MODEL'
        st.session_state.training_df = pd.DataFrame()

if load_model_button:
    log_and_display_message("Caricamento del modello esistente...", "info")
    st.session_state.model, st.session_state.tokenizer = jg.load_fine_tuned_model(OUTPUT_DIR, log_and_display_message)
    if st.session_state.model and st.session_state.tokenizer:
        st.session_state.model_state = 'READY'
    else:
        st.session_state.model_state = 'NO_MODEL'

# ==============================================================================
# SEZIONE 2.2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione Giudizi")

if st.session_state.model_state == 'READY':
    excel_file_to_complete = st.file_uploader(
        "Carica file Excel da completare", 
        type=['xlsx', 'xls', 'xlsm'], 
        key='excel_file_to_complete_uploader'
    )
    
    # Se viene caricato un file per la generazione, mostra il dropdown e il pulsante
    if excel_file_to_complete:
        st.session_state.file_uploaded_for_generation = True
        
        # Leggi i nomi dei fogli di lavoro
        st.session_state.sheets_for_generation = er.get_excel_sheet_names(BytesIO(excel_file_to_complete.getvalue()))
        st.session_state.selected_sheet = st.selectbox(
            "Seleziona Foglio di Lavoro da completare", 
            st.session_state.sheets_for_generation,
            key='sheet_dropdown'
        )

        process_excel_button = st.button(
            "Avvia Generazione su File", 
            key='process_excel_button'
        )

        if process_excel_button and st.session_state.selected_sheet:
            log_and_display_message("Avvio della generazione dei giudizi sul file caricato...", "info")
            
            # Leggi solo il foglio selezionato
            excel_df = pd.read_excel(BytesIO(excel_file_to_complete.getvalue()), sheet_name=st.session_state.selected_sheet)
            
            # Trova l'ultima riga completata
            giudizio_col = er.find_giudizio_column_name(excel_df.columns, log_and_display_message)
            start_index = get_last_completed_row(excel_df, giudizio_col)
            
            st.session_state.processed_rows = start_index
            st.session_state.total_rows = len(excel_df)
            
            # Utilizza una barra di progresso per l'utente
            progress_bar = st.progress(0, text=f"Generazione in corso... ({st.session_state.processed_rows}/{st.session_state.total_rows})")

            # Genera i giudizi
            st.session_state.process_completed_file = jg.generate_judgments_for_excel(
                model=st.session_state.model,
                tokenizer=st.session_state.tokenizer,
                excel_df=excel_df,
                start_row=start_index,
                sheet_name=st.session_state.selected_sheet,
                progress_container=log_and_display_message,
                progress_bar=progress_bar
            )

    else:
        log_and_display_message("Carica un file Excel e seleziona il foglio di lavoro per avviare la generazione.", "info")

else:
    log_and_display_message("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.", "warning")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.header("3. Stato e Download")

# Visualizza i messaggi di stato
for message in st.session_state.status_messages:
    if message['type'] == "error":
        st.error(message['message'])
    elif message['type'] == "warning":
        st.warning(message['message'])
    else:
        st.info(message['message'])
        
if st.session_state.process_completed_file is not None:
    st.write("### Scarica il file completato")
    
    # Creiamo un buffer in memoria per il file Excel
    output_buffer = BytesIO()
    with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
        st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
    output_buffer.seek(0)
    
    st.download_button(
        label="Scarica il file aggiornato",
        data=output_buffer,
        file_name=f"giudizi_completo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

