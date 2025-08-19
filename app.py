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

# Importa i moduli personalizzati
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Configurazione della pagina di Streamlit
st.set_page_config(layout="wide", page_title="Generatore di Giudizi AI")

# Definizione delle directory per il salvataggio del modello e dei dati
OUTPUT_DIR = "modello_finetunato"

# ==============================================================================
# SEZIONE 1: GESTIONE DELLO STATO DELLA SESSIONE E PULIZIA
# ==============================================================================
# Inizializza le variabili di sessione se non esistono
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'model_state' not in st.session_state:
    st.session_state.model_state = 'ready' # può essere 'ready', 'training', 'generating'
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

def reset_session_state():
    """Resetta lo stato della sessione per una nuova operazione."""
    st.session_state.status_messages = []
    st.session_state.process_completed_file = None

# Aggiungi un pulsante per resettare lo stato
if st.button("Reset Sessione"):
    reset_session_state()
    st.rerun()

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE E FLUSSO DI LAVORO
# ==============================================================================
st.title("Generatore di Giudizi AI")

# Contenitore per i messaggi di stato
status_container = st.empty()

# Funzione per aggiungere messaggi di stato
def append_status(message, container):
    st.session_state.status_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    with container:
        st.info(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    st.session_state.process_completed_file = None

# --------------------
# 2.1 Fine-Tuning del modello
# --------------------
st.header("1. Addestramento del Modello (Fine-Tuning)")
st.markdown("Carica un file Excel con le colonne 'Giudizio' e le informazioni che compongono il giudizio (es. 'Valutazione', 'Dettaglio', 'Punteggio'). Il modello apprenderà dagli esempi.")

fine_tune_file = st.file_uploader(
    "Carica file di addestramento",
    type=['xlsx', 'xls', 'xlsm'],
    key="fine_tune_uploader",
    help="Carica il file Excel per addestrare il modello."
)

col1, col2 = st.columns(2)
with col1:
    start_fine_tune = st.button("Avvia Fine-Tuning", disabled=(st.session_state.model_state != 'ready' or fine_tune_file is None), use_container_width=True)
with col2:
    delete_corpus_button = st.button("Elimina Corpus", disabled=(st.session_state.model_state != 'ready'), use_container_width=True)

if start_fine_tune:
    st.session_state.model_state = 'training'
    with st.spinner("Preparazione dati e avvio addestramento..."):
        try:
            # 1. Legge il file di addestramento
            df_for_training = er.read_excel_file_to_df(fine_tune_file, st.session_state.status_messages)
            if df_for_training.empty:
                st.error("Nessun dato valido trovato per l'addestramento. Controlla il file.")
            else:
                # 2. Aggiorna il corpus esistente con i nuovi dati
                corpus_df = cb.build_or_update_corpus(df_for_training, st.session_state.status_messages)
                
                # 3. Avvia il fine-tuning solo se il corpus non è vuoto
                if not corpus_df.empty:
                    mt.fine_tune_model(corpus_df, OUTPUT_DIR, st.session_state.status_messages)
                    st.success("Addestramento completato con successo!")
                    st.balloons()
                else:
                    st.error("Corpus vuoto. Impossibile avviare l'addestramento.")

        except Exception as e:
            st.error(f"Errore durante il fine-tuning: {e}\n\nTraceback:\n{traceback.format_exc()}")
        finally:
            st.session_state.model_state = 'ready'
            st.rerun()

if delete_corpus_button:
    with st.spinner("Eliminazione del corpus..."):
        cb.delete_corpus(st.session_state.status_messages)
        st.rerun()

# --------------------
# 2.2 Generazione dei giudizi
# --------------------
st.markdown("---")
st.header("2. Generazione dei Giudizi su File")
st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna e potrai scaricare il file aggiornato.")

excel_file = st.file_uploader(
    "Carica file Excel da completare",
    type=['xlsx', 'xls', 'xlsm'],
    key="excel_uploader",
    help="Carica il file Excel in cui generare i giudizi."
)

# Carica il modello pre-addestrato per verificare la sua presenza
model, tokenizer = jg.load_trained_model(OUTPUT_DIR)
model_is_ready = model is not None and tokenizer is not None
if not model_is_ready:
    st.warning("Nessun modello addestrato trovato. Esegui prima il Fine-Tuning.")

if excel_file is not None:
    excel_df = pd.read_excel(excel_file)
    sheets = pd.ExcelFile(excel_file).sheet_names
    st.session_state.selected_sheet = st.selectbox("Seleziona Foglio di Lavoro", sheets, index=0)

    start_generation = st.button("Avvia Generazione", disabled=(st.session_state.model_state != 'ready' or not model_is_ready or excel_file is None), use_container_width=True)

    if start_generation:
        st.session_state.model_state = 'generating'
        with st.spinner("Avvio generazione giudizi..."):
            try:
                # Carica il DataFrame del foglio selezionato
                df_to_complete = pd.read_excel(excel_file, sheet_name=st.session_state.selected_sheet)
                
                # Verifica che la colonna "Giudizio" esista (case-insensitive)
                giudizio_col = er.find_giudizio_column(df_to_complete)
                
                if giudizio_col:
                    # Avvia la generazione
                    completed_df = jg.generate_judgments_for_excel(
                        model, tokenizer, df_to_complete, giudizio_col, st.session_state.selected_sheet, OUTPUT_DIR, st.session_state.status_messages
                    )
                    st.session_state.process_completed_file = completed_df
                    st.success("Generazione completata con successo!")
                    st.balloons()
                else:
                    st.error("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che l'intestazione esista.")

            except Exception as e:
                st.error(f"Errore nella lettura del file Excel: {e}\n\nTraceback:\n{traceback.format_exc()}")
            finally:
                st.session_state.model_state = 'ready'
                st.rerun()
                
# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.header("3. Stato e Download")

# Visualizza i messaggi di stato
for message in st.session_state.status_messages:
    if "Errore" in message:
        st.error(message)
    elif "Attenzione" in message:
        st.warning(message)
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
        file_name=f"Giudizi_Completati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
