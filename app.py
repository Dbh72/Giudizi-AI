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

def progress_container(message, status="info"):
    """
    Gestisce i messaggi di stato in un contenitore Streamlit.
    """
    if status == "info":
        st.info(message)
    elif status == "success":
        st.success(message)
    elif status == "warning":
        st.warning(message)
    elif status == "error":
        st.error(message)

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE E LOGICA
# ==============================================================================
st.set_page_config(page_title="Generatore Automatico di Giudizi", layout="wide")
st.title("Sistema di Generazione Giudizi con AI")

# Inizializza le variabili di sessione
if 'uploaded_training_file' not in st.session_state:
    st.session_state.uploaded_training_file = None
if 'uploaded_process_file' not in st.session_state:
    st.session_state.uploaded_process_file = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'is_model_trained' not in st.session_state:
    st.session_state.is_model_trained = os.path.exists(os.path.join(OUTPUT_DIR, "final_model"))
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.markdown("---")
st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel con giudizi già completati per costruire il corpus di addestramento e/o addestrare un nuovo modello. Se hai già un modello fine-tunato, puoi saltare questo passaggio.")

# Carica corpus esistente all'avvio
if st.session_state.corpus_df.empty:
    st.session_state.corpus_df = cb.load_corpus(progress_container)

with st.expander("Gestione del Corpus di Addestramento"):
    st.write("Puoi caricare uno o più file per costruire il corpus. Il corpus verrà salvato per usi futuri.")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_training_file = st.file_uploader("Carica File di Addestramento (.xlsx)", type=["xlsx"], key="training_file_uploader")
    with col2:
        if st.session_state.uploaded_training_file:
            training_sheet = st.selectbox("Seleziona Foglio di Lavoro", er.get_excel_sheet_names(st.session_state.uploaded_training_file))
            if st.button("Aggiorna Corpus"):
                training_df = er.read_excel_for_training(st.session_state.uploaded_training_file, [training_sheet], progress_container)
                st.session_state.corpus_df = cb.build_or_update_corpus(training_df, progress_container)
    
    if st.button("Visualizza Corpus Attuale"):
        if not st.session_state.corpus_df.empty:
            st.dataframe(st.session_state.corpus_df.head(10))
            st.write(f"Totale righe nel corpus: {len(st.session_state.corpus_df)}")
        else:
            st.warning("Il corpus di addestramento è vuoto.")
    
    if st.button("Elimina Corpus Esistente"):
        cb.delete_corpus(progress_container)
        st.session_state.corpus_df = pd.DataFrame()

# Logica di addestramento
if st.session_state.is_model_trained:
    st.success("Un modello addestrato è già presente.")
    if st.button("Carica modello per la generazione"):
        st.session_state.model, st.session_state.tokenizer = mt.load_fine_tuned_model(progress_container)
    if st.button("Elimina modello addestrato"):
        mt.delete_model(progress_container)
        st.session_state.is_model_trained = False
        st.session_state.model = None
        st.session_state.tokenizer = None
else:
    st.warning("Nessun modello addestrato trovato.")
    if st.session_state.corpus_df.empty:
        st.error("Il corpus di addestramento è vuoto. Carica un file per addestrare il modello.")
    else:
        if st.button("Avvia Fine-Tuning del Modello"):
            st.session_state.model, st.session_state.tokenizer = mt.fine_tune_model(st.session_state.corpus_df, progress_container)
            if st.session_state.model and st.session_state.tokenizer:
                st.session_state.is_model_trained = True
                progress_container("Addestramento completato e modello caricato con successo!", "success")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione Giudizi su File")
st.markdown("Carica un file Excel con le colonne 'Materia' e 'Descrizione Giudizio' per generare i giudizi.")
uploaded_process_file = st.file_uploader("Carica File Excel da Processare (.xlsx)", type=["xlsx"], key="process_file_uploader")

if uploaded_process_file:
    st.session_state.uploaded_process_file = uploaded_process_file
    sheet_names = er.get_excel_sheet_names(uploaded_process_file)
    st.session_state.selected_sheet = st.selectbox("Seleziona Foglio di Lavoro", sheet_names, key="process_sheet_selector")

if st.session_state.is_model_trained and st.session_state.uploaded_process_file:
    if st.button("Avvia Generazione Giudizi"):
        if st.session_state.model is None or st.session_state.tokenizer is None:
            st.session_state.model, st.session_state.tokenizer = mt.load_fine_tuned_model(progress_container)
        if st.session_state.model and st.session_state.tokenizer:
            try:
                processed_df = er.read_excel_for_generation(st.session_state.uploaded_process_file, st.session_state.selected_sheet, progress_container)
                if not processed_df.empty:
                    st.session_state.process_completed_file = jg.generate_judgments(processed_df, st.session_state.model, st.session_state.tokenizer, st.session_state.selected_sheet, progress_container)
                    st.success("Generazione dei giudizi completata!")
                else:
                    st.error("Nessun dato valido trovato nel foglio selezionato.")
            except Exception as e:
                progress_container(f"Errore nella generazione dei giudizi: {e}", "error")
                progress_container(f"Traceback: {traceback.format_exc()}", "error")
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

