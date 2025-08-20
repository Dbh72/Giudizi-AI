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

# Funzione per registrare i messaggi di progresso e stato nell'interfaccia utente
def progress_container(message, type="info"):
    """
    Mostra un messaggio di stato colorato nell'interfaccia Streamlit.
    """
    if type == "info":
        st.info(message)
    elif type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE PRINCIPALE
# ==============================================================================

st.set_page_config(layout="wide", page_title="Generatore Automatico di Giudizi")

st.title("Generatore Automatico di Giudizi 🤖")
st.markdown("Questa applicazione ti aiuta a generare giudizi automaticamente utilizzando un modello di linguaggio fine-tuned.")

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.markdown("---")
st.header("1. Addestramento del Modello")
st.markdown("Qui puoi addestrare un nuovo modello o aggiornarne uno esistente con un corpus di dati personalizzato.")

# Creiamo contenitori per i messaggi di stato
status_placeholder_train = st.empty()

with st.expander("Carica Dati di Addestramento"):
    uploaded_train_file = st.file_uploader("Carica un file Excel per l'addestramento", type=['xlsx'], help="Carica un file contenente i dati di addestramento. La colonna 'Giudizio' è obbligatoria.")
    
    # Se un file è stato caricato, mostra le opzioni
    if uploaded_train_file is not None:
        sheet_names = er.get_excel_sheet_names(uploaded_train_file)
        selected_sheets = st.multiselect("Seleziona i fogli da utilizzare per l'addestramento:", options=sheet_names, default=sheet_names)
    
        if st.button("Avvia Aggiornamento Corpus", help="Avvia l'aggiornamento del corpus di addestramento"):
            if selected_sheets:
                try:
                    df_corpus = er.read_and_prepare_data_from_excel(uploaded_train_file, selected_sheets, progress_container)
                    if not df_corpus.empty:
                        updated_corpus = cb.build_or_update_corpus(df_corpus, progress_container)
                        st.session_state.corpus_df = updated_corpus
                except Exception as e:
                    progress_container(f"Errore nella preparazione dei dati di addestramento: {e}", "error")
                    st.error("Si è verificato un errore durante l'aggiornamento del corpus. Controlla il formato del file.")
            else:
                progress_container("Per favore, seleziona almeno un foglio da elaborare.", "warning")

with st.expander("Opzioni di Addestramento Avanzate"):
    st.markdown("Qui puoi gestire il modello esistente o il corpus.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Elimina Corpus di Addestramento", help="Cancella il corpus di dati precedentemente salvato."):
            cb.delete_corpus(progress_container)
            if 'corpus_df' in st.session_state:
                del st.session_state['corpus_df']

    with col2:
        if st.button("Elimina Modello Fine-Tuned", help="Cancella il modello addestrato precedentemente salvato."):
            mt.delete_model(progress_container)
            if 'model' in st.session_state:
                del st.session_state['model']
            if 'tokenizer' in st.session_state:
                del st.session_state['tokenizer']

if st.button("Avvia Addestramento del Modello", help="Avvia il fine-tuning del modello con il corpus di addestramento."):
    if 'corpus_df' in st.session_state and not st.session_state.corpus_df.empty:
        model, tokenizer = mt.train_model(st.session_state.corpus_df, progress_container)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        progress_container("Addestramento completato e modello caricato in memoria!", "success")
    else:
        progress_container("Corpus non trovato o vuoto. Carica un file di addestramento per iniziare.", "warning")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione dei Giudizi")
st.markdown("Carica il file Excel che vuoi completare e genera i giudizi mancanti.")

# Carica il modello pre-addestrato se esiste
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = mt.load_fine_tuned_model(progress_container)
    if st.session_state.model is not None:
        progress_container("Modello caricato con successo!", "success")

status_placeholder_generate = st.empty()

if st.session_state.model is not None and st.session_state.tokenizer is not None:
    uploaded_process_file = st.file_uploader("Carica il file Excel da processare", type=['xlsx'], help="Carica il file che contiene le righe da completare con i giudizi.")
    
    if uploaded_process_file is not None:
        sheet_names_process = er.get_excel_sheet_names(uploaded_process_file)
        selected_sheet = st.selectbox("Seleziona il foglio da completare:", options=sheet_names_process)
        
        if st.button("Genera Giudizi", help="Avvia il processo di generazione per il foglio selezionato."):
            try:
                processed_df = er.read_and_prepare_data_from_excel(uploaded_process_file, [selected_sheet], progress_container, training_mode=False)
                if not processed_df.empty:
                    st.session_state.process_completed_file = jg.generate_judgments_and_save(processed_df, st.session_state.model, st.session_state.tokenizer, selected_sheet, progress_container)
                else:
                    progress_container("Il file caricato non contiene dati validi. Controlla il formato.", "error")
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

if st.session_state.get('process_completed_file') is not None:
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
