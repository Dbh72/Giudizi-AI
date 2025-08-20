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
def progress_container(placeholder, message, type):
    """
    Gestisce la visualizzazione dei messaggi di stato.
    - placeholder: il contenitore Streamlit in cui scrivere il messaggio.
    - message: il testo del messaggio.
    - type: 'info', 'success', 'warning', 'error'.
    """
    if placeholder:
        if type == 'info':
            placeholder.info(message)
        elif type == 'success':
            placeholder.success(message)
        elif type == 'warning':
            placeholder.warning(message)
        elif type == 'error':
            placeholder.error(message)

# Funzione per l'addestramento del modello
def train_model(corpus_df, progress_placeholder):
    progress_container(progress_placeholder, "Avvio dell'addestramento del modello...", "info")
    try:
        model, tokenizer = mt.train_fine_tuned_model(corpus_df, progress_container=lambda msg, type: progress_container(progress_placeholder, msg, type))
        if model and tokenizer:
            st.session_state.trained_model = model
            st.session_state.trained_tokenizer = tokenizer
            progress_container(progress_placeholder, "Addestramento completato e modello caricato con successo!", "success")
        else:
            progress_container(progress_placeholder, "Addestramento fallito. Controlla i log per i dettagli.", "error")
    except Exception as e:
        progress_container(progress_placeholder, f"Errore durante l'addestramento: {e}", "error")
        progress_container(progress_placeholder, f"Traceback: {traceback.format_exc()}", "error")


# Funzione per caricare un modello esistente
def load_existing_model(progress_placeholder):
    progress_container(progress_placeholder, "Caricamento del modello esistente...", "info")
    try:
        model, tokenizer = mt.load_fine_tuned_model(progress_container=lambda msg, type: progress_container(progress_placeholder, msg, type))
        if model and tokenizer:
            st.session_state.trained_model = model
            st.session_state.trained_tokenizer = tokenizer
            progress_container(progress_placeholder, "Modello caricato con successo!", "success")
        else:
            progress_container(progress_placeholder, "Modello non trovato. Devi addestrarne uno nuovo.", "warning")
    except Exception as e:
        progress_container(progress_placeholder, f"Errore nel caricamento del modello: {e}", "error")
        progress_container(progress_placeholder, f"Traceback: {traceback.format_exc()}", "error")

# Funzione per la generazione dei giudizi e salvataggio
def generate_judgments_and_save(file_object, sheet_name, progress_placeholder):
    progress_container(progress_placeholder, f"Preparazione per la generazione dei giudizi nel foglio '{sheet_name}'...", "info")
    try:
        # Carica il dataframe dal file
        df = er.read_and_prepare_data_from_excel(file_object, [sheet_name], progress_container=lambda msg, type: progress_container(progress_placeholder, msg, type))

        if not df.empty:
            progress_container(progress_placeholder, "Avvio della generazione dei giudizi...", "info")
            
            # Chiamata alla funzione corretta nel modulo judgment_generator
            updated_df = jg.generate_judgments(df, st.session_state.trained_model, st.session_state.trained_tokenizer, sheet_name, progress_container=lambda msg, type: progress_container(progress_placeholder, msg, type))
            
            st.session_state.process_completed_file = updated_df
            st.session_state.selected_sheet = sheet_name
            progress_container(progress_placeholder, "Generazione dei giudizi completata con successo! Ora puoi scaricare il file.", "success")
        else:
            progress_container(progress_placeholder, f"Nessun dato valido trovato nel foglio '{sheet_name}'.", "warning")
            st.session_state.process_completed_file = None
            st.session_state.selected_sheet = None

    except Exception as e:
        progress_container(progress_placeholder, f"Errore durante la generazione dei giudizi: {e}", "error")
        progress_container(progress_placeholder, f"Traceback: {traceback.format_exc()}", "error")
        st.error(f"Errore durante la generazione dei giudizi: {e}")

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE CON STREAMLIT
# ==============================================================================
st.set_page_config(page_title="Giudizi-AI", page_icon="🤖", layout="wide")

st.title("🤖 Giudizi-AI: Il tuo Assistente per la Valutazione")
st.markdown("Questa applicazione ti aiuta a generare automaticamente giudizi per le verifiche scolastiche, partendo dai tuoi file Excel.")

# Inizializzazione degli stati di sessione
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_tokenizer' not in st.session_state:
    st.session_state.trained_tokenizer = None
if 'processed_file_data' not in st.session_state:
    st.session_state.processed_file_data = None
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = ""

st.sidebar.header("Pannello di Controllo")
if st.sidebar.button("Carica Modello Esistente"):
    load_existing_model(st.sidebar.empty())

if st.sidebar.button("Elimina Modello Addestrato"):
    mt.delete_model(st.sidebar.empty())
    st.session_state.trained_model = None
    st.session_state.trained_tokenizer = None
    st.sidebar.success("Modello eliminato.")

if st.sidebar.button("Carica Corpus Esistente"):
    st.session_state.corpus_df = cb.load_corpus(st.sidebar.empty())

if st.sidebar.button("Elimina Corpus"):
    cb.delete_corpus(st.sidebar.empty())
    st.session_state.corpus_df = pd.DataFrame()
    st.sidebar.success("Corpus eliminato.")


# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.header("1. Addestramento del Modello")
st.info("Carica i tuoi file Excel per creare un corpus di addestramento e addestrare il modello. Questo è il primo passo.")
uploaded_train_file = st.file_uploader("Carica file Excel per l'addestramento", type=['xlsx', 'xls', 'xlsm'], key="train_file")

if uploaded_train_file is not None:
    st.session_state.uploaded_train_file = uploaded_train_file
    status_placeholder_train = st.empty()
    if st.button("Avvia Addestramento", key="train_button"):
        with st.spinner("Preparazione dati..."):
            try:
                # Legge il file caricato e prepara il dataframe
                new_data_df = er.read_and_prepare_data_from_excel(
                    st.session_state.uploaded_train_file,
                    er.get_excel_sheet_names(st.session_state.uploaded_train_file),
                    progress_container=lambda msg, type: progress_container(status_placeholder_train, msg, type)
                )

                if not new_data_df.empty:
                    # Aggiorna il corpus esistente con i nuovi dati
                    st.session_state.corpus_df = cb.build_or_update_corpus(
                        new_data_df,
                        progress_container=lambda msg, type: progress_container(status_placeholder_train, msg, type)
                    )
                    
                    if not st.session_state.corpus_df.empty:
                        train_model(st.session_state.corpus_df, status_placeholder_train)
                    else:
                        progress_container(status_placeholder_train, "Corpus vuoto. Impossibile avviare l'addestramento.", "error")
                else:
                    progress_container(status_placeholder_train, "Nessun dato valido trovato nel file per l'addestramento.", "warning")

            except Exception as e:
                progress_container(status_placeholder_train, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
                st.error("Errore nel caricamento del file. Controlla il formato e riprova.")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione dei Giudizi")
st.info("Carica il file Excel da completare con i giudizi. Assicurati che il modello sia stato addestrato.")

if st.session_state.trained_model is not None and st.session_state.trained_tokenizer is not None:
    uploaded_process_file = st.file_uploader("Carica file Excel da processare", type=['xlsx', 'xls', 'xlsm'], key="process_file")

    if uploaded_process_file is not None:
        st.session_state.uploaded_process_file = uploaded_process_file
        sheets = er.get_excel_sheet_names(uploaded_process_file)
        selected_sheet = st.selectbox("Seleziona il foglio da processare", options=sheets)
        
        status_placeholder_generate = st.empty()
        
        if st.button("Genera Giudizi", key="generate_button"):
            try:
                generate_judgments_and_save(st.session_state.uploaded_process_file, selected_sheet, status_placeholder_generate)
            except Exception as e:
                progress_container(status_placeholder_generate, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
                st.error("Errore nel caricamento del file. Controlla il formato e riprova.")
    else:
        st.warning("Per generare i giudizi, devi prima caricare un file nella sezione '2. Generazione dei Giudizi'.")
else:
    st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.header("3. Stato e Download")

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

