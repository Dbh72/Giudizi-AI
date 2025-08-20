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

# Funzione per registrare i messaggi di progresso e stato.
def progress_container(placeholder, message, status_type="info"):
    """Visualizza un messaggio di stato in un placeholder."""
    # Definisce le icone e i colori per i diversi tipi di stato
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    # Mostra il messaggio con l'icona e il colore appropriati
    if status_type == "success":
        placeholder.success(f"{icons.get(status_type)} {message}")
    elif status_type == "warning":
        placeholder.warning(f"{icons.get(status_type)} {message}")
    elif status_type == "error":
        placeholder.error(f"{icons.get(status_type)} {message}")
    else:
        placeholder.info(f"{icons.get(status_type)} {message}")


# Funzione per addestrare il modello.
def train_model_and_save(file, sheet, status_placeholder):
    """
    Gestisce l'intero processo di addestramento del modello.
    1. Legge il file Excel.
    2. Costruisce/aggiorna il corpus.
    3. Avvia il fine-tuning del modello.
    """
    with st.spinner("Caricamento e preparazione dati..."):
        # Legge il file Excel
        progress_container(status_placeholder, "Avvio del processo di addestramento...")
        df = er.read_excel_file_for_corpus(file, sheet, status_placeholder)
        if df.empty:
            progress_container(status_placeholder, "Nessun dato valido nel file di addestramento.", "error")
            return

    with st.spinner("Creazione o aggiornamento del corpus di addestramento..."):
        # Costruisce/aggiorna il corpus
        corpus_df = cb.build_or_update_corpus(df, lambda msg, type: progress_container(status_placeholder, msg, type))
        if corpus_df.empty:
            return

    with st.spinner("Avvio del fine-tuning del modello..."):
        # Addestra il modello
        mt.fine_tune_model(corpus_df, lambda msg, type: progress_container(status_placeholder, msg, type))
        st.session_state.model_trained = True


# Funzione per generare i giudizi.
def generate_judgments_and_save(file, sheet, status_placeholder):
    """
    Gestisce l'intero processo di generazione dei giudizi.
    1. Carica il modello fine-tuned.
    2. Legge il file Excel da completare.
    3. Genera i giudizi.
    4. Salva il file completato.
    """
    with st.spinner("Caricamento del modello e del tokenizer..."):
        # Carica il modello
        model, tokenizer = mt.load_fine_tuned_model(lambda msg, type: progress_container(status_placeholder, msg, type))
        if model is None or tokenizer is None:
            progress_container(status_placeholder, "Errore nel caricamento del modello. Controlla che il modello sia stato addestrato.", "error")
            st.session_state.process_completed_file = None
            return

    with st.spinner("Lettura del file da elaborare..."):
        # Legge il file da elaborare
        df = er.read_excel_file_for_processing(file, sheet, status_placeholder)
        if df.empty:
            st.session_state.process_completed_file = None
            return
            
    with st.spinner("Generazione dei giudizi in corso..."):
        # Genera i giudizi
        completed_df = jg.generate_judgments(df, model, tokenizer, sheet, lambda msg, type: progress_container(status_placeholder, msg, type))
        if completed_df is not None:
            st.session_state.process_completed_file = completed_df
            progress_container(status_placeholder, "Giudizi generati con successo!", "success")
            
# Funzione per cancellare il file del corpus e del modello
def delete_all_data():
    """Cancella tutti i dati di addestramento e il modello."""
    cb.delete_corpus(lambda msg, type: st.info(msg))
    mt.delete_model(lambda msg, type: st.info(msg))
    st.session_state.model_trained = False
    st.session_state.process_completed_file = None
    st.session_state.uploaded_train_file = None
    st.session_state.uploaded_process_file = None
    st.rerun()

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE PRINCIPALE (STREAMLIT)
# ==============================================================================
st.set_page_config(page_title="Generatore di Giudizi AI", layout="wide")

st.title("Generatore di Giudizi per Aziende")
st.markdown("---")
st.markdown("Questa applicazione ti guida attraverso l'addestramento di un modello di linguaggio per generare giudizi, basato sui tuoi dati storici.")

# Inizializza lo stato della sessione
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = mt.model_exists()
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'uploaded_train_file' not in st.session_state:
    st.session_state.uploaded_train_file = None
if 'uploaded_process_file' not in st.session_state:
    st.session_state.uploaded_process_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# Aggiunge un pulsante per il reset dell'applicazione
if st.button("Reset Applicazione e Cancella Dati"):
    delete_all_data()

st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel (`.xlsx` o `.xlsm`) contenente le colonne `Prompt` e `Giudizio` per addestrare il modello.")

# Gestione del file di addestramento
uploaded_train_file = st.file_uploader(
    "Carica il file di addestramento:",
    type=["xlsx", "xlsm"],
    key="train_file_uploader"
)

# Gestione del reset del caricatore di file
if st.button("Ricarica File di Addestramento"):
    st.session_state.uploaded_train_file = None
    st.session_state.selected_sheet = None
    st.rerun()

if uploaded_train_file:
    st.session_state.uploaded_train_file = uploaded_train_file
    st.session_state.selected_sheet = None
    
    sheet_names = er.get_excel_sheet_names(uploaded_train_file)
    if not sheet_names:
        st.error("Errore: Impossibile leggere i fogli di lavoro dal file.")
    else:
        st.session_state.selected_sheet = st.selectbox(
            "Seleziona il foglio di lavoro da utilizzare:",
            sheet_names
        )

# Bottone per addestrare il modello
if st.button("Avvia Addestramento", key="train_button"):
    if st.session_state.uploaded_train_file and st.session_state.selected_sheet:
        status_placeholder_train = st.empty()
        try:
            train_model_and_save(st.session_state.uploaded_train_file, st.session_state.selected_sheet, status_placeholder_train)
        except Exception as e:
            progress_container(status_placeholder_train, f"Errore durante l'addestramento. Controlla il formato del file e riprova. {e}", "error")
            st.error("Errore durante l'addestramento. Controlla il formato del file e riprova.")
    else:
        st.warning("Devi caricare un file Excel di addestramento e selezionare un foglio per iniziare.")

st.markdown("---")
st.header("2. Generazione dei Giudizi")
st.markdown("Carica un nuovo file Excel con la colonna `Prompt` per generare i giudizi. Assicurati di aver prima addestrato il modello.")

# Gestione del file da processare
uploaded_process_file = st.file_uploader(
    "Carica il file da completare:",
    type=["xlsx", "xlsm"],
    key="process_file_uploader"
)

# Gestione del reset del caricatore di file
if st.button("Ricarica File da Completare"):
    st.session_state.uploaded_process_file = None
    st.rerun()

if uploaded_process_file:
    st.session_state.uploaded_process_file = uploaded_process_file

# Bottone per generare i giudizi
if st.button("Genera Giudizi", key="generate_button"):
    if st.session_state.model_trained:
        if st.session_state.uploaded_process_file and st.session_state.selected_sheet:
            status_placeholder_generate = st.empty()
            try:
                generate_judgments_and_save(st.session_state.uploaded_process_file, st.session_state.selected_sheet, status_placeholder_generate)
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

