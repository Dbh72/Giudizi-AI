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

# Funzione per registrare i messaggi di stato.
def progress_container(message, type="info"):
    """Aggiunge un messaggio alla session state per la visualizzazione e forza il refresh."""
    st.session_state.status_messages.append({"message": message, "type": type})
    st.rerun()

# Funzione per la pulizia dello stato dopo l'uso.
def clear_state():
    """Pulisce le variabili di stato per un nuovo processo."""
    st.session_state.status_messages = []
    if "corpus_df" in st.session_state:
        del st.session_state.corpus_df
    if "trained_model_exists" in st.session_state:
        del st.session_state.trained_model_exists
    if "process_completed_file" in st.session_state:
        del st.session_state.process_completed_file
    if "selected_sheet" in st.session_state:
        del st.session_state.selected_sheet
    
# Funzione per inizializzare o caricare lo stato.
def init_state():
    """Inizializza le variabili di sessione di Streamlit."""
    if "status_messages" not in st.session_state:
        st.session_state.status_messages = []
    if "corpus_df" not in st.session_state:
        st.session_state.corpus_df = cb.load_corpus(progress_container)
    if "trained_model_exists" not in st.session_state:
        st.session_state.trained_model_exists = os.path.exists(os.path.join(OUTPUT_DIR, "final_model"))
    if "process_completed_file" not in st.session_state:
        st.session_state.process_completed_file = None
    if "selected_sheet" not in st.session_state:
        st.session_state.selected_sheet = None

# Funzione per l'addestramento del modello.
def train_model_and_save(train_file):
    """Gestisce il flusso di addestramento del modello."""
    clear_state()
    try:
        df = er.read_and_prepare_data_from_excel(train_file.name, [er.get_excel_sheet_names(train_file.name)[0]], progress_container)
        st.session_state.corpus_df = cb.build_or_update_corpus(df, progress_container)
        
        if not st.session_state.corpus_df.empty:
            mt.fine_tune_model(st.session_state.corpus_df, progress_container)
            st.session_state.trained_model_exists = True
    except Exception as e:
        progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")

# Funzione per la generazione dei giudizi.
def generate_judgments_and_save(process_file, sheet_name):
    """Gestisce il flusso di generazione dei giudizi."""
    clear_state()
    progress_container(f"Generazione dei giudizi per il foglio '{sheet_name}'...", "info")
    try:
        model, tokenizer = jg.load_model(os.path.join(OUTPUT_DIR, "final_model"), progress_container)
        if model and tokenizer:
            processed_df = jg.generate_judgments_for_excel(
                model=model,
                tokenizer=tokenizer,
                file_path=process_file.name,
                sheet_name=sheet_name,
                progress_container=progress_container
            )
            st.session_state.process_completed_file = processed_df
            st.session_state.selected_sheet = sheet_name
            progress_container("Processo completato!", "success")
        else:
            progress_container("Impossibile caricare il modello. Assicurati che il fine-tuning sia stato completato con successo.", "error")
    except Exception as e:
        progress_container(f"Errore durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")


# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE
# ==============================================================================

# Impostazioni generali
st.set_page_config(
    page_title="Generatore Automatico di Giudizi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Generatore Automatico di Giudizi per Docenti")
st.markdown("---")
st.markdown("Questa applicazione utilizza un modello di linguaggio per generare automaticamente giudizi a partire dai dati di un file Excel.")
st.markdown("---")

init_state()

# Cache the model loading to prevent it from reloading on every interaction.
# The old st.experimental_singleton has been replaced with st.cache_resource.
# This decorator ensures the function is run only once for a given set of arguments.
# Since we're not passing any arguments, it will be run just one time on app startup.
@st.cache_resource
def load_cached_model():
    """
    Carica il modello e il tokenizer solo una volta per tutta la sessione.
    Utilizza st.cache_resource per un caching efficiente di oggetti complessi.
    """
    if st.session_state.trained_model_exists:
        try:
            return jg.load_model(os.path.join(OUTPUT_DIR, "final_model"), progress_container)
        except Exception as e:
            progress_container(f"Errore nel caricamento del modello cache: {e}", "error")
            return None, None
    return None, None

# Main Sections
st.header("1. Addestramento del Modello")
st.markdown("Carica qui un file Excel con i giudizi per addestrare il modello.")

with st.expander("Dettagli per l'addestramento"):
    st.info("Il modello imparerà a generare giudizi in base ai dati forniti.")
    st.markdown("Assicurati che il tuo file Excel contenga una colonna denominata **'Giudizio'** con i giudizi completi e le altre colonne con i dati da cui il modello dovrà apprendere.")
    st.markdown("Puoi caricare anche più fogli nello stesso file, verranno tutti aggregati per l'addestramento.")

uploaded_train_file = st.file_uploader(
    "Carica File Excel per l'addestramento",
    type=["xlsx", "xls", "xlsm"],
    help="Seleziona il file Excel contenente i dati di addestramento."
)

if uploaded_train_file:
    if st.button("Avvia Addestramento"):
        train_model_and_save(uploaded_train_file)

if st.session_state.trained_model_exists:
    st.success("Modello addestrato trovato e pronto all'uso!")

    if st.button("Elimina modello addestrato"):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            st.session_state.trained_model_exists = False
            clear_state()
            st.rerun()
        else:
            progress_container("Nessun modello da eliminare.", "warning")

st.markdown("---")

st.header("2. Generazione dei Giudizi")
st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna e potrai scaricare il file aggiornato.")

if st.session_state.trained_model_exists:
    uploaded_process_file = st.file_uploader(
        "Carica File Excel da completare",
        type=["xlsx", "xls", "xlsm"],
        help="Seleziona il file Excel per la generazione dei giudizi."
    )

    if uploaded_process_file:
        try:
            sheet_names = er.get_excel_sheet_names(uploaded_process_file.name)
            selected_sheet = st.selectbox(
                "Seleziona il Foglio di Lavoro",
                options=sheet_names,
                index=0
            )

            if st.button("Avvia Generazione"):
                generate_judgments_and_save(uploaded_process_file, selected_sheet)
        except Exception as e:
            progress_container(f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
            st.error("Errore nel caricamento del file. Controlla il formato e riprova.")
else:
    st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

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
        file_name=f"Giudizi_Generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

