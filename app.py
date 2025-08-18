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
CORPUS_FILE = "training_corpus.parquet"

# Inizializza lo stato della sessione
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.st.session_state:
    st.session_state.process_completed_file = None
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None

def reset_state():
    """Resetta lo stato della sessione per un nuovo processo."""
    st.session_state.status_messages = []
    st.session_state.process_completed_file = None

def get_current_status_message():
    """Ottiene il messaggio di stato corrente."""
    return st.session_state.status_messages[-1] if st.session_state.status_messages else ""

# Creazione di un container per i messaggi di stato
status_container = st.container()

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.title("Generatore di Giudizi AI 🤖")
st.markdown("---")
st.header("1. Addestramento del Modello")
st.markdown("Carica un file per l'addestramento. Formati supportati: Excel, PDF, DOCX, TXT.")

training_file = st.file_uploader("Carica file per l'addestramento", type=["xlsx", "xls", "pdf", "docx", "doc", "txt"], key="training_uploader")

col1, col2 = st.columns(2)
with col1:
    train_button = st.button("Avvia Addestramento", help="Avvia l'addestramento (fine-tuning) del modello.")
with col2:
    delete_corpus_button = st.button("Elimina Corpus", help="Elimina il corpus di addestramento salvato.")

if delete_corpus_button:
    reset_state()
    st.session_state.status_messages.append("Inizializzazione...")
    er.delete_corpus(st.session_state.status_messages)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        st.session_state.status_messages.append("Modello fine-tunato eliminato.")
    st.session_state.model_ready = False
    st.session_state.current_model_path = None
    st.session_state.status_messages.append("Corpus e modello eliminati. Pronto per un nuovo addestramento.")

if train_button and training_file:
    reset_state()
    st.session_state.status_messages.append("Inizializzazione del processo di addestramento...")
    
    # Crea o aggiorna il corpus
    corpus_df = er.load_and_update_corpus(training_file, st.session_state.status_messages)
    
    if not corpus_df.empty:
        try:
            st.session_state.status_messages.append("Corpus di addestramento pronto. Avvio fine-tuning...")
            
            # Utilizza il corpus aggiornato per l'addestramento
            final_model_path = mt.fine_tune_model(corpus_df, st.session_state.status_messages)
            
            if final_model_path:
                st.session_state.current_model_path = final_model_path
                st.session_state.model_ready = True
                st.session_state.status_messages.append("Addestramento completato. Modello pronto per la generazione!")
                st.balloons()
            else:
                st.session_state.status_messages.append("Errore: Addestramento non completato.")
                st.session_state.model_ready = False
        except Exception as e:
            st.session_state.status_messages.append(f"Errore critico durante l'addestramento: {e}")
            st.session_state.status_messages.append(traceback.format_exc())
            st.session_state.model_ready = False
            
elif train_button and not training_file:
    st.error("Per favore, carica un file per l'addestramento.")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione dei Giudizi")
st.markdown("Carica un file Excel con la colonna 'Giudizio' da completare.")

judgment_file = st.file_uploader("Carica file Excel per la generazione", type=["xlsx", "xls"], key="judgment_uploader")

if st.session_state.model_ready and judgment_file:
    # Controlla il tipo di file
    if not isinstance(judgment_file, BytesIO):
        st.error("Si prega di caricare un file Excel valido.")
        judgment_file = BytesIO(judgment_file.getvalue())

    try:
        # Trova il nome del foglio di lavoro
        with pd.ExcelFile(judgment_file) as xls:
            sheet_names = xls.sheet_names
        
        if len(sheet_names) > 1:
            st.session_state.selected_sheet = st.selectbox("Seleziona il foglio di lavoro", sheet_names, help="Seleziona il foglio che contiene i dati per la generazione dei giudizi.")
        else:
            st.session_state.selected_sheet = sheet_names[0]
            st.write(f"Foglio di lavoro selezionato: **{st.session_state.selected_sheet}**")

        if st.button("Genera Giudizi", key="generate_button"):
            reset_state()
            st.session_state.status_messages.append("Inizializzazione del processo di generazione...")

            # Carica il modello e il tokenizer
            model, tokenizer = jg.load_trained_model(st.session_state.current_model_path)
            
            if model and tokenizer:
                st.session_state.status_messages.append("Modello e tokenizer caricati con successo.")
                
                # Prepara il DataFrame da completare
                df_to_complete, giudizio_col = er.prepare_dataframe_for_generation(judgment_file, st.session_state.selected_sheet)
                
                if df_to_complete is not None and giudizio_col is not None:
                    st.session_state.status_messages.append("DataFrame pronto per la generazione.")
                    
                    # Genera i giudizi
                    completed_df = jg.generate_judgments_for_excel(
                        model,
                        tokenizer,
                        df_to_complete,
                        giudizio_col,
                        st.session_state.selected_sheet,
                        st.session_state.current_model_path
                    )
                    
                    st.session_state.process_completed_file = completed_df
                    st.success("Generazione completata con successo!")
                    st.balloons()
                else:
                    st.error("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che l'intestazione esista.")
            else:
                st.error("Errore nel caricamento del modello. Assicurati che il percorso sia corretto.")

    except Exception as e:
        st.error(f"Errore nella lettura del file Excel: {e}\n\nTraceback:\n{traceback.format_exc()}")

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
        file_name=f"Giudizi_generati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

