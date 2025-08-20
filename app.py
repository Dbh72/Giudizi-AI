# ==============================================================================
# File: app.py
# L'interfaccia utente principale per l'applicazione di generazione di giudizi.
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
import time

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

# Funzione per registrare i messaggi di progresso e stato
def progress_container(message, message_type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = ""
    if message_type == "info":
        formatted_message = f"[{timestamp}] INFO: {message}"
    elif message_type == "success":
        formatted_message = f"[{timestamp}] SUCCESSO: {message}"
    elif message_type == "warning":
        formatted_message = f"[{timestamp}] AVVISO: {message}"
    elif message_type == "error":
        formatted_message = f"[{timestamp}] ERRORE: {message}"

    st.session_state.log_messages.append(formatted_message)
    st.session_state.log_messages = st.session_state.log_messages[-50:] # Limita il log


# Funzione per eseguire il fine-tuning e caricare il modello nella sessione
def perform_training(uploaded_train_file):
    try:
        st.session_state.process_completed_file = None
        st.session_state.selected_sheet = None

        file_content = uploaded_train_file.getvalue()
        sheet_names = er.get_excel_sheet_names(BytesIO(file_content))
        
        # CHIAMATA CORRETTA: Rimosso l'argomento 'sheets_to_ignore' che non è più necessario
        # e che causava l'errore. La logica di filtraggio è ora in excel_reader.
        new_data_df = er.read_and_prepare_data_from_excel(BytesIO(file_content), sheet_names, progress_container)
        
        if new_data_df.empty:
            progress_container("Nessun dato valido trovato per l'addestramento. Addestramento annullato.", "error")
            return

        st.session_state.corpus = cb.build_or_update_corpus(new_data_df, progress_container)
        
        model, tokenizer = mt.fine_tune_model(st.session_state.corpus, progress_container)
        
        if model is not None and tokenizer is not None:
            st.session_state.trained_model = model
            st.session_state.trained_tokenizer = tokenizer
            progress_container("Modello addestrato e pronto per la generazione.", "success")
        else:
            progress_container("Addestramento fallito.", "error")

    except Exception as e:
        progress_container(f"Errore durante l'addestramento: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")

# Funzione per generare i giudizi e salvare il file
def generate_judgments_and_save(uploaded_process_file, selected_sheet):
    try:
        if st.session_state.trained_model is None or st.session_state.trained_tokenizer is None:
            progress_container("Modello non caricato. Carica il modello nella sezione '1. Addestramento del Modello'.", "error")
            return
            
        file_content = uploaded_process_file.getvalue()
        
        file_object_io = BytesIO(file_content)
        df_to_process = pd.read_excel(file_object_io, sheet_name=selected_sheet, header=None)
        
        df_to_process, giudizio_col_name = er.find_header_row_and_columns(df_to_process, progress_container)

        if df_to_process is None or giudizio_col_name is None:
            progress_container("Errore durante la preparazione del file.", "error")
            return
            
        completed_df = jg.generate_judgments_on_dataframe(df_to_process, selected_sheet, st.session_state.trained_model, st.session_state.trained_tokenizer, giudizio_col_name, progress_container)
        
        st.session_state.process_completed_file = completed_df
        st.session_state.selected_sheet = selected_sheet
        progress_container("File completato e pronto per il download.", "success")

    except Exception as e:
        progress_container(f"Errore durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE (STREAMLIT)
# ==============================================================================

st.set_page_config(layout="wide", page_title="Giudizi-AI", page_icon="🤖")

st.title("🤖 Giudizi-AI: Il tuo Assistente per la Valutazione")
st.markdown("Questa applicazione ti aiuta a generare automaticamente giudizi per le verifiche scolastiche, partendo dai tuoi file Excel.")

# Inizializzazione degli stati di sessione
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_tokenizer' not in st.session_state:
    st.session_state.trained_tokenizer = None
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame()
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Addestramento del Modello")
    st.markdown("Carica un file Excel con esempi di giudizi per addestrare o aggiornare il modello.")

    uploaded_train_file = st.file_uploader(
        "Carica file di addestramento (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=False,
        key="uploader_train"
    )

    if uploaded_train_file:
        if st.button("Addestra il modello", key="train_button"):
            perform_training(uploaded_train_file)
    
    st.markdown("---")
    st.header("2. Generazione dei Giudizi")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota per completarla.")

    if st.session_state.trained_model:
        uploaded_process_file = st.file_uploader(
            "Carica file da processare (.xlsx, .xls, .xlsm)",
            type=["xlsx", "xls", "xlsm"],
            accept_multiple_files=False,
            key="uploader_process"
        )

        if uploaded_process_file:
            try:
                file_content = uploaded_process_file.getvalue()
                sheet_names = er.get_excel_sheet_names(BytesIO(file_content))
                
                selected_sheet = st.selectbox(
                    "Seleziona il foglio di lavoro da elaborare:",
                    options=sheet_names,
                    key="sheet_selector"
                )
                
                if st.button("Genera Giudizi", key="generate_button"):
                    generate_judgments_and_save(uploaded_process_file, selected_sheet)

            except Exception as e:
                progress_container(f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
    else:
        st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

with col2:
    st.header("3. Stato e Download")
    st.markdown("Qui puoi vedere lo stato delle operazioni e scaricare il file completato.")
    st.markdown("---")

    st.subheader("Log delle operazioni")
    log_container = st.container()
    with log_container:
        for msg in reversed(st.session_state.log_messages):
            if "[INFO]" in msg:
                st.info(msg)
            elif "[SUCCESSO]" in msg:
                st.success(msg)
            elif "[AVVISO]" in msg:
                st.warning(msg)
            elif "[ERRORE]" in msg:
                st.error(msg)
            else:
                st.write(msg)
    
    st.markdown("---")
    
    if st.session_state.process_completed_file is not None:
        st.subheader("Scarica il file completato")
        
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
        output_buffer.seek(0)
        
        st.download_button(
            label="Scarica il file aggiornato",
            data=output_buffer,
            file_name=f"Giudizi_Generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_button"
        )
    else:
        st.info("Il file aggiornato sarà disponibile qui dopo la generazione.")

    st.markdown("---")
    
    st.subheader("Gestione")
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        if st.button("Elimina Corpus", key="delete_corpus_button"):
            cb.delete_corpus(progress_container)
            st.session_state.corpus = pd.DataFrame()
            st.session_state.process_completed_file = None

    with col_del2:
        if st.button("Elimina Modello", key="delete_model_button"):
            mt.delete_model(progress_container)
            st.session_state.trained_model = None
            st.session_state.trained_tokenizer = None

