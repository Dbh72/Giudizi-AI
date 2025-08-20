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

def get_progress_container(placeholder):
    """Restituisce una funzione per aggiornare un container di stato."""
    def progress_container(message, type="info"):
        if type == "info":
            placeholder.info(message)
        elif type == "success":
            placeholder.success(message)
        elif type == "error":
            placeholder.error(message)
        elif type == "warning":
            placeholder.warning(message)
        time.sleep(1) # Un piccolo delay per la visualizzazione
    return progress_container

def fine_tune_and_save(uploaded_file, selected_sheets, status_placeholder_train):
    """Gestisce il flusso di addestramento del modello e il salvataggio."""
    progress_container = get_progress_container(status_placeholder_train)
    
    # Costruisci/aggiorna il corpus
    corpus_df = er.read_training_file(uploaded_file, selected_sheets, progress_container)
    if corpus_df.empty:
        progress_container("Addestramento annullato: Il file non contiene dati validi per l'addestramento.", "error")
        st.session_state.model_trained = False
        return
        
    updated_corpus = cb.build_or_update_corpus(corpus_df, progress_container)
    
    if updated_corpus.empty:
        progress_container("Addestramento annullato: Impossibile creare o aggiornare il corpus.", "error")
        st.session_state.model_trained = False
        return

    # Addestra il modello
    model, tokenizer = mt.fine_tune_model(updated_corpus, progress_container)
    
    if model is not None and tokenizer is not None:
        st.session_state.model_trained = True
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        progress_container("Modello addestrato con successo!", "success")
        
def generate_judgments_and_save(uploaded_file, selected_sheet, status_placeholder_generate):
    """Gestisce il flusso di generazione dei giudizi e il salvataggio."""
    progress_container = get_progress_container(status_placeholder_generate)
    
    # Carica il modello e il tokenizer
    model, tokenizer = mt.load_fine_tuned_model(progress_container)
    
    if model is None or tokenizer is None:
        progress_container("Generazione annullata: Modello non disponibile. Addestra un modello prima di procedere.", "error")
        st.session_state.process_completed_file = None
        st.session_state.selected_sheet = None
        return
    
    # Leggi il file da processare
    df, giudizio_col_name, input_cols_name = er.read_excel_file(uploaded_file, selected_sheet, progress_container)
    
    if df.empty:
        progress_container("Generazione annullata: Il file non contiene dati validi per la generazione.", "error")
        st.session_state.process_completed_file = None
        st.session_state.selected_sheet = None
        return
    
    # Genera i giudizi
    processed_df = jg.generate_judgments(df, model, tokenizer, giudizio_col_name, input_cols_name, selected_sheet, progress_container)
    
    st.session_state.process_completed_file = processed_df
    st.session_state.selected_sheet = selected_sheet
    progress_container("Processo di generazione completato. Puoi scaricare il file aggiornato.", "success")
    
# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE STREAMLIT
# ==============================================================================

# Inizializzazione degli stati della sessione
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

st.set_page_config(page_title="Generatore di Giudizi per Aziende", layout="wide")
st.title("Generatore di Giudizi Aziendali con AI")
st.markdown("---")

# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel con esempi di giudizi per addestrare il modello. **Assicurati che il file abbia una colonna 'Giudizio' e le colonne di input (es. 'Testo', 'Descrizione').**")

uploaded_training_file = st.file_uploader(
    "Carica un file Excel per l'addestramento",
    type=["xlsx", "xls"],
    key="training_uploader"
)

# SEZIONE per la gestione del corpus
st.markdown("### Gestione Corpus")
status_placeholder_corpus = st.empty()

if st.button("Carica/Aggiorna Corpus"):
    if uploaded_training_file:
        try:
            progress_container = get_progress_container(status_placeholder_corpus)
            sheet_names = er.get_excel_sheet_names(uploaded_training_file)
            corpus_df = er.read_training_file(uploaded_training_file, sheet_names, progress_container)
            if not corpus_df.empty:
                cb.build_or_update_corpus(corpus_df, progress_container)
                st.session_state.model_trained = False # Invalida il modello precedente
        except Exception as e:
            progress_container("Errore nel caricamento del file. Controlla il formato e riprova.", "error")
            st.error(f"Errore: {e}")
    else:
        status_placeholder_corpus.warning("Per favore, carica un file prima di procedere.")

if st.button("Elimina Corpus e Modello"):
    progress_container = get_progress_container(status_placeholder_corpus)
    cb.delete_corpus(progress_container)
    mt.delete_model(progress_container)
    st.session_state.model_trained = False

st.markdown("---")

# SEZIONE per l'addestramento
status_placeholder_train = st.empty()
if st.button("Avvia Addestramento Modello"):
    corpus_df = cb.load_corpus(get_progress_container(status_placeholder_train))
    if not corpus_df.empty:
        try:
            fine_tune_and_save(uploaded_training_file, er.get_excel_sheet_names(uploaded_training_file), status_placeholder_train)
        except Exception as e:
            get_progress_container(status_placeholder_train)(f"Errore durante l'addestramento: {e}", "error")
    else:
        status_placeholder_train.warning("Per favore, carica un file di addestramento e aggiorna il corpus prima di addestrare il modello.")

st.markdown("---")

# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.header("2. Generazione Giudizi")
st.markdown("Carica un file Excel senza la colonna 'Giudizio' per generarla automaticamente.")

uploaded_process_file = st.file_uploader(
    "Carica il file Excel da processare",
    type=["xlsx", "xls"],
    key="process_uploader"
)

if uploaded_process_file:
    sheet_names = er.get_excel_sheet_names(uploaded_process_file)
    selected_sheet = st.selectbox(
        "Seleziona il foglio di lavoro da processare",
        sheet_names
    )
    
    status_placeholder_generate = st.empty()
    if st.button("Avvia Generazione Giudizi"):
        if st.session_state.model_trained or os.path.exists(os.path.join(OUTPUT_DIR, "final_model")):
            try:
                generate_judgments_and_save(uploaded_process_file, selected_sheet, status_placeholder_generate)
            except Exception as e:
                progress_container(status_placeholder_generate, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
                st.error("Errore nel caricamento del file. Controlla il formato e riprova.")
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

