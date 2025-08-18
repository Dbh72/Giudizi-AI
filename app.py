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

# Importa i moduli personalizzati
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Configurazione della pagina di Streamlit
st.set_page_config(layout="wide", page_title="Generatore di Giudizi AI")

# Definizione delle directory per il salvataggio del modello e dei dati
OUTPUT_DIR = "modello_finetunato"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Inizializza lo stato della sessione di Streamlit
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'df_to_complete' not in st.session_state:
    st.session_state.df_to_complete = None
if 'loaded_model' not in st.session_state:
    st.session_state.loaded_model = None
if 'loaded_tokenizer' not in st.session_state:
    st.session_state.loaded_tokenizer = None

# ==============================================================================
# SEZIONE 1: FUNZIONI CALLBACK E LOGICA APPLICATIVA
# ==============================================================================

@st.cache_resource
def get_trained_model():
    """
    Carica il modello addestrato in cache per evitare di ricaricarlo ogni volta.
    """
    if st.session_state.loaded_model is None or st.session_state.loaded_tokenizer is None:
        model, tokenizer = jg.load_trained_model(OUTPUT_DIR)
        st.session_state.loaded_model = model
        st.session_state.loaded_tokenizer = tokenizer
    return st.session_state.loaded_model, st.session_state.loaded_tokenizer

def train_model_callback(df_train):
    """
    Funzione callback per l'addestramento del modello.
    """
    st.session_state.status_messages.append("Preparazione per l'addestramento...")
    st.session_state.status_messages.append(f"Avvio addestramento con {len(df_train)} esempi.")
    
    try:
        mt.train_model(df_train, st.session_state.status_messages, OUTPUT_DIR)
        st.session_state.status_messages.append("Addestramento completato con successo!")
        st.balloons()
        # Ricarica il modello appena addestrato nella sessione
        st.session_state.loaded_model = None
        st.session_state.loaded_tokenizer = None
        get_trained_model()
    except Exception as e:
        st.session_state.status_messages.append(f"Errore durante l'addestramento: {e}")
        st.session_state.status_messages.append(f"Traceback:\n{traceback.format_exc()}")
    st.rerun()

def generate_judgments_callback(df_to_complete, giudizio_col, selected_sheet):
    """
    Funzione callback per la generazione dei giudizi.
    """
    if st.session_state.loaded_model and st.session_state.loaded_tokenizer:
        try:
            completed_df = jg.generate_judgments_for_excel(
                st.session_state.loaded_model,
                st.session_state.loaded_tokenizer,
                df_to_complete,
                giudizio_col,
                selected_sheet,
                OUTPUT_DIR,
                st.session_state.status_messages
            )
            st.session_state.process_completed_file = completed_df
            st.session_state.status_messages.append("Generazione completata con successo!")
            st.balloons()
        except Exception as e:
            st.session_state.status_messages.append(f"Errore durante l'operazione: {e}\n\nTraceback:\n{traceback.format_exc()}")
    else:
        st.session_state.status_messages.append("Errore: Modello non caricato. Addestra o carica un modello.")
    st.rerun()

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE DI STREAMLIT
# ==============================================================================

st.title("Generatore di Giudizi per Excel con AI")
st.subheader("Una soluzione per automatizzare la compilazione dei giudizi in base alle descrizioni.")

st.markdown("---")

# SEZIONE 2.1: CARICAMENTO MODELLO O FILE DI ADDESTRAMENTO
# ==============================================================================
st.header("1. Addestra il Modello o Carica uno esistente")
col1, col2 = st.columns(2)

# Caricamento del file di addestramento (sezione sinistra)
with col1:
    st.subheader("Addestra un nuovo modello")
    uploaded_train_file = st.file_uploader(
        "Carica il file Excel per l'addestramento (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="uploader_train"
    )

    if uploaded_train_file:
        df_train = er.prepare_training_data(uploaded_train_file)
        if not df_train.empty:
            st.success(f"File di addestramento caricato con successo. Trovati {len(df_train)} esempi.")
            st.write("Anteprima del dataset:")
            st.dataframe(df_train.head())
            if st.button("Avvia Addestramento", key="train_button"):
                train_model_callback(df_train)
        else:
            st.warning("Nessun dato valido trovato nel file di addestramento.")

# Caricamento del modello esistente (sezione destra)
with col2:
    st.subheader("Carica un modello esistente")
    st.info("Il modello verrà caricato automaticamente dalla directory 'modello_finetunato' se esiste.")
    
    if st.button("Carica Modello", key="load_model_button"):
        with st.spinner("Caricamento del modello e del tokenizer..."):
            model, tokenizer = get_trained_model()
            if model and tokenizer:
                st.success("Modello e tokenizer caricati con successo!")
            else:
                st.error("Errore: Impossibile caricare il modello. Assicurati che esista la directory 'modello_finetunato' con il modello salvato.")
    
    if st.session_state.loaded_model:
        st.success("Modello e tokenizer già caricati in memoria.")

st.markdown("---")

# SEZIONE 2.2: CARICAMENTO FILE DA COMPLETARE E GENERAZIONE
# ==============================================================================
st.header("2. Genera Giudizi su un File Esistente")
uploaded_file_to_complete = st.file_uploader(
    "Carica il file Excel da completare (.xlsx, .xls, .xlsm)",
    type=["xlsx", "xls", "xlsm"],
    key="uploader_complete"
)

if uploaded_file_to_complete:
    # Mostra i fogli di lavoro disponibili
    try:
        excel_file = pd.ExcelFile(uploaded_file_to_complete)
        sheet_names = excel_file.sheet_names
        st.session_state.selected_sheet = st.selectbox("Seleziona il foglio di lavoro da completare:", sheet_names)

        # Prepara il DataFrame da completare
        df_to_complete, giudizio_col = er.prepare_dataframe_to_complete(uploaded_file_to_complete, st.session_state.selected_sheet)
        
        if df_to_complete is not None and giudizio_col:
            st.session_state.df_to_complete = df_to_complete
            st.success(f"Foglio '{st.session_state.selected_sheet}' caricato con successo.")
            st.info(f"Verranno completate le righe mancanti nella colonna '{giudizio_col}'.")
            st.write("Anteprima del file da completare:")
            st.dataframe(st.session_state.df_to_complete.head())
            
            if st.button(f"Avvia Generazione su '{st.session_state.selected_sheet}'", key="generate_button"):
                generate_judgments_callback(st.session_state.df_to_complete, giudizio_col, st.session_state.selected_sheet)
        elif not giudizio_col:
            st.error("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che l'intestazione esista.")
        else:
            st.error("Errore nel caricamento del file. Controlla il formato e i dati.")

    except Exception as e:
        st.error(f"Errore nella lettura del file Excel: {e}\n\nTraceback:\n{traceback.format_exc()}")
        
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.header("3. Stato e Download")

status_container = st.container()
with status_container:
    for message in st.session_state.status_messages:
        if "Errore" in message:
            st.error(message)
        else:
            st.info(message)

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
        file_name=f"Giudizi_generati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_button"
    )

