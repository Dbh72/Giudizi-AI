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
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, CORPUS_FILE, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================
def initialize_state():
    """
    Inizializza le variabili di stato della sessione di Streamlit.
    """
    if "status_messages" not in st.session_state:
        st.session_state.status_messages = []
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "process_completed_file" not in st.session_state:
        st.session_state.process_completed_file = None
    if "selected_sheet" not in st.session_state:
        st.session_state.selected_sheet = None
    if "model_ready" not in st.session_state:
        st.session_state.model_ready = False
    if "model" not in st.session_state:
        st.session_state.model = None
    if "tokenizer" not in st.session_state:
        st.session_state.tokenizer = None
    if "start_row" not in st.session_state:
        st.session_state.start_row = 0

def progress_container(message, type="info"):
    """
    Aggiunge un messaggio di stato alla lista da visualizzare.
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.status_messages.append({"message": f"[{timestamp}] {message}", "type": type})
    st.experimental_rerun()

def delete_existing_model():
    """
    Elimina il modello e il tokenizer salvati localmente.
    """
    progress_container("Tentativo di eliminazione del modello esistente...", "info")
    if os.path.exists(OUTPUT_DIR):
        try:
            shutil.rmtree(OUTPUT_DIR)
            progress_container(f"Modello in '{OUTPUT_DIR}' eliminato con successo.", "success")
        except OSError as e:
            progress_container(f"Errore: {e.strerror} - Impossibile eliminare la directory {e.filename}", "error")
    else:
        progress_container("Nessun modello esistente da eliminare.", "warning")

def fine_tune_model(progress_container, fine_tune_file):
    """
    Funzione per avviare il fine-tuning del modello.
    """
    if fine_tune_file is None:
        progress_container("Errore: Seleziona un file Excel per l'addestramento.", "error")
        return

    progress_container("Avvio del processo di addestramento...", "info")
    
    # Prepara il dataframe per il corpus
    df_corpus = er.read_and_prepare_data_from_excel(fine_tune_file.name, progress_container, None)
    
    if df_corpus.empty:
        progress_container("Errore: Il file di addestramento non contiene dati validi.", "error")
        return

    # Costruisci o aggiorna il corpus
    cb.build_or_update_corpus(df_corpus, progress_container)

    # Carica il corpus per l'addestramento
    corpus_df = cb.load_corpus(progress_container)

    if corpus_df.empty:
        progress_container("Errore: Nessun corpus di addestramento disponibile.", "error")
        return

    # Avvia l'addestramento del modello
    try:
        mt.train_model(corpus_df, progress_container)
        st.session_state.model_trained = True
        progress_container("Addestramento completato con successo!", "success")
        st.experimental_rerun()
    except Exception as e:
        progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")

def get_excel_sheet_names_and_enable_button(excel_file):
    """
    Carica i nomi dei fogli di un file Excel e abilita i controlli UI.
    """
    if excel_file is not None:
        progress_container("File caricato. Lettura dei fogli...", "info")
        try:
            sheet_names = er.get_excel_sheet_names(excel_file.name)
            st.session_state.sheet_names = sheet_names
            return sheet_names, False
        except Exception as e:
            progress_container(f"Errore nella lettura dei fogli di lavoro: {e}", "error")
            return [], True
    return [], True

def generate_judgments_on_excel(excel_file, selected_sheet, progress_container):
    """
    Genera i giudizi per le righe mancanti nel file Excel.
    """
    if excel_file is None or selected_sheet is None:
        progress_container("Errore: Seleziona un file e un foglio di lavoro.", "error")
        return
    
    if not st.session_state.model_ready:
        progress_container("Errore: Il modello non è ancora pronto. Attendi il completamento del caricamento.", "error")
        return

    progress_container(f"Avvio della generazione per il file: {excel_file.name}, Foglio: {selected_sheet}...", "info")

    try:
        # Carica il dataframe dal file Excel e foglio specificato
        df_to_complete = er.read_and_prepare_data_from_excel(
            excel_file.name,
            progress_container,
            [selected_sheet],
            "complete"
        )
        
        if df_to_complete.empty:
            progress_container("Attenzione: Il foglio selezionato non contiene dati validi. Nessun giudizio generato.", "warning")
            st.session_state.process_completed_file = None
            return

        # Rileva le righe già compilate e l'ultima riga elaborata
        start_row_index, df_with_judgments = jg.find_last_processed_row(df_to_complete)
        st.session_state.start_row = start_row_index
        
        progress_container(f"Trovate {len(df_to_complete)} righe totali. Verrà ripreso dall'indice: {start_row_index}.", "info")

        # Genera i giudizi per le righe mancanti
        df_completed = jg.generate_judgments(
            st.session_state.model,
            st.session_state.tokenizer,
            df_with_judgments,
            progress_container,
            start_row_index
        )
        
        st.session_state.process_completed_file = df_completed
        st.session_state.selected_sheet = selected_sheet
        progress_container("Generazione dei giudizi completata. Il file è pronto per il download.", "success")
        st.experimental_rerun()

    except Exception as e:
        progress_container(f"Errore durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        st.session_state.process_completed_file = None

def load_model_on_init():
    """
    Carica il modello fine-tuned all'avvio dell'applicazione.
    """
    if not st.session_state.model_ready:
        progress_container("Caricamento del modello fine-tuned. Attendere...", "info")
        model, tokenizer = jg.load_finetuned_model(OUTPUT_DIR, progress_container)
        
        if model and tokenizer:
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.model_ready = True
            st.session_state.model_trained = True
            progress_container("Modello pronto per la generazione di giudizi.", "success")
        else:
            st.session_state.model_ready = False
            st.session_state.model_trained = False
            progress_container("Nessun modello fine-tuned trovato. Addestra un modello per abilitare la generazione.", "warning")

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE DI STREAMLIT
# ==============================================================================

# Inizializzazione dello stato all'inizio della sessione
initialize_state()

# Caricamento del modello una tantum all'avvio
if not st.session_state.model_ready:
    st.experimental_singleton(load_model_on_init)()

st.set_page_config(
    page_title="Generatore Giudizi Scolastici",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍🏫 Generatore Giudizi Scolastici con IA")
st.markdown("---")

# Area di visualizzazione dei messaggi di stato
st.header("Stato dell'Operazione")
status_placeholder = st.empty()
with status_placeholder.container():
    for message in st.session_state.status_messages:
        if message['type'] == "error":
            st.error(message['message'])
        elif message['type'] == "warning":
            st.warning(message['message'])
        else:
            st.info(message['message'])
st.markdown("---")

# Sezione 1: Addestramento del Modello (Fine-Tuning)
st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel per aggiornare il corpus di addestramento e fare il fine-tuning del modello.")
fine_tune_file = st.file_uploader(
    "Carica file Excel per l'addestramento",
    type=["xlsx", "xls", "xlsm"],
    key="fine_tune_uploader",
    help="Il file deve contenere una colonna 'Giudizio' e le colonne 'input' per il modello."
)
col1, col2 = st.columns(2)
with col1:
    train_button = st.button(
        "Avvia Addestramento", 
        disabled=fine_tune_file is None,
        help="Avvia il fine-tuning del modello."
    )
with col2:
    delete_corpus_button = st.button(
        "Cancella Corpus",
        help="Elimina il corpus di addestramento esistente."
    )

if train_button:
    st.session_state.status_messages = []
    fine_tune_model(progress_container, fine_tune_file)

if delete_corpus_button:
    st.session_state.status_messages = []
    cb.delete_corpus(progress_container)
    st.session_state.model_trained = False

# Sezione 2: Generazione Giudizi
st.header("2. Generazione Giudizi su File")

if st.session_state.model_trained:
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna.")
    process_excel_file = st.file_uploader(
        "Carica file Excel da completare",
        type=["xlsx", "xls", "xlsm"],
        key="process_excel_uploader",
        help="Il file deve avere le stesse colonne del file di addestramento e una colonna 'Giudizio'."
    )
    
    sheet_names_to_display = []
    if process_excel_file is not None:
        try:
            sheet_names_to_display = er.get_excel_sheet_names(process_excel_file.name)
            if not sheet_names_to_display:
                st.warning("Nessun foglio di lavoro trovato nel file.")
        except Exception as e:
            st.error("Errore nella lettura dei fogli di lavoro. Controlla il file.")
            progress_container(f"Errore: {e}", "error")

    selected_sheet = st.selectbox(
        "Seleziona il Foglio di Lavoro",
        options=sheet_names_to_display,
        disabled=not sheet_names_to_display
    )
    
    process_excel_button = st.button(
        "Avvia Generazione",
        disabled=(process_excel_file is None or selected_sheet is None)
    )

    if process_excel_button:
        st.session_state.status_messages = []
        generate_judgments_on_excel(process_excel_file, selected_sheet, progress_container)

    # Sezione di download
    if st.session_state.process_completed_file is not None:
        st.markdown("---")
        st.header("3. Scarica il file completato")
        
        # Creiamo un buffer in memoria per il file Excel
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Assicurati che il dataframe sia scritto nel foglio corretto
            st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
        output_buffer.seek(0)
        
        st.download_button(
            label="Scarica il file aggiornato",
            data=output_buffer,
            file_name=f"giudizi_completati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

