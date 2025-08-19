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
from datasets import Dataset, DatasetDict
import re
import time

# Importa i moduli personalizzati e il file di configurazione
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Configurazione della pagina di Streamlit
st.set_page_config(layout="wide", page_title="Generatore di Giudizi AI")

# Inizializzazione delle variabili di stato di Streamlit
if 'model_state' not in st.session_state:
    st.session_state.model_state = 'not_loaded'
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'excel_file_path' not in st.session_state:
    st.session_state.excel_file_path = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'corpus_exists' not in st.session_state:
    st.session_state.corpus_exists = False
if 'model_exists' not in st.session_state:
    st.session_state.model_exists = False

# ==============================================================================
# SEZIONE 1: FUNZIONI PER LA GESTIONE DEGLI STATI E DEL FLUSSO
# ==============================================================================

def update_status(message, type="info"):
    """Aggiunge un messaggio di stato alla lista."""
    st.session_state.status_messages.append(message)

def clear_status():
    """Cancella tutti i messaggi di stato."""
    st.session_state.status_messages = []

def check_file_states():
    """Controlla l'esistenza dei file principali (corpus e modello) all'avvio."""
    # Controlla se il corpus di addestramento esiste
    if os.path.exists(cb.CORPUS_FILE):
        st.session_state.corpus_exists = True
        update_status("Corpus di addestramento trovato.", "info")
    else:
        st.session_state.corpus_exists = False
        update_status("Nessun corpus di addestramento trovato. Devi caricarne uno per addestrare il modello.", "warning")
    
    # Controlla se il modello fine-tuned esiste
    if os.path.exists(OUTPUT_DIR):
        st.session_state.model_exists = True
        update_status(f"Modello fine-tuned trovato in '{OUTPUT_DIR}'. Puoi passare direttamente alla sezione 2.", "info")
    else:
        st.session_state.model_exists = False
        update_status("Nessun modello fine-tuned trovato. Devi addestrare il modello prima di generare giudizi.", "warning")

def fine_tune_model_flow(file_input):
    """Gestisce l'intero flusso di fine-tuning del modello."""
    clear_status()
    st.session_state.model_state = 'training'
    
    update_status("Avvio del processo di fine-tuning...", "info")
    
    try:
        if file_input:
            # Step 1: Aggiorna o costruisce il corpus
            update_status("Aggiornamento del corpus con i dati del file Excel...", "info")
            corpus_df = er.read_excel_file_to_df(BytesIO(file_input.getvalue()), update_status)
            if not corpus_df.empty:
                corpus_df = cb.build_or_update_corpus(corpus_df, update_status)
            
            # Step 2: Avvia l'addestramento
            if not corpus_df.empty:
                update_status(f"Avvio dell'addestramento del modello con {len(corpus_df)} righe di dati...", "info")
                model_trainer_status = mt.fine_tune(corpus_df, update_status)
                if model_trainer_status:
                    update_status("Fine-tuning completato con successo!", "success")
                else:
                    update_status("Errore durante il fine-tuning. Controlla i log.", "error")
            else:
                update_status("Il file caricato non contiene dati validi per l'addestramento.", "error")
        else:
            update_status("Per favore, carica un file Excel con i dati di addestramento.", "warning")
            
    except Exception as e:
        update_status(f"Errore critico durante il fine-tuning: {e}\n\nTraceback:\n{traceback.format_exc()}", "error")
    finally:
        st.session_state.model_state = 'ready'
        check_file_states()

def process_excel_for_judgments():
    """Gestisce il flusso di generazione dei giudizi su un file Excel."""
    clear_status()
    st.session_state.model_state = 'generating'
    
    if st.session_state.excel_file_path and st.session_state.selected_sheet:
        update_status(f"Avvio della generazione dei giudizi per il foglio '{st.session_state.selected_sheet}'...", "info")
        
        try:
            # Carica il modello fine-tuned
            model, tokenizer = jg.load_trained_model(OUTPUT_DIR, update_status)
            
            if model and tokenizer:
                # Legge il file Excel da completare
                df_to_complete = er.read_excel_file_to_df(st.session_state.excel_file_path, update_status, sheet_name=st.session_state.selected_sheet, read_only=True)
                
                if not df_to_complete.empty:
                    # Trova la colonna 'Giudizio'
                    giudizio_col = er.find_giudizio_column(df_to_complete)
                    if giudizio_col:
                        # Genera i giudizi
                        update_status(f"Trovata la colonna 'Giudizio'. Generazione in corso...", "info")
                        completed_df = jg.generate_judgments_for_excel(
                            model, tokenizer, df_to_complete, giudizio_col, st.session_state.selected_sheet, OUTPUT_DIR, update_status
                        )
                        st.session_state.process_completed_file = completed_df
                        update_status("Generazione completata con successo!", "success")
                        st.balloons()
                    else:
                        update_status("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che l'intestazione esista.", "error")
                else:
                    update_status("Errore nel caricamento del file. Controlla il formato e i dati.", "error")
            else:
                update_status("Errore nel caricamento del modello. Assicurati che il percorso sia corretto e che il modello sia stato addestrato.", "error")

        except Exception as e:
            update_status(f"Errore nella lettura o nella generazione: {e}\n\nTraceback:\n{traceback.format_exc()}", "error")
        finally:
            st.session_state.model_state = 'ready'
            st.rerun()

# ==============================================================================
# SEZIONE 2: LAYOUT DELL'INTERFACCIA UTENTE
# ==============================================================================

# Titolo principale
st.title("Generatore di Giudizi AI")
st.markdown("---")

# Controlla lo stato dei file all'avvio o dopo un'operazione
if 'first_run' not in st.session_state:
    st.session_state.first_run = True
    check_file_states()

st.header("1. Addestramento Incrementale del Modello")
st.markdown("Carica un file Excel (con colonne 'Giudizio' e 'descrizione') per addestrare il modello. Il processo riprenderà dall'ultimo stato salvato.")
st.markdown(f"**Stato Corpus:** {'Presente' if st.session_state.corpus_exists else 'Non Presente'}")
st.markdown(f"**Stato Modello:** {'Presente' if st.session_state.model_exists else 'Non Presente'}")

with st.container(border=True):
    fine_tune_file_input = st.file_uploader("Carica file Excel per l'addestramento", type=['xlsx', 'xls', 'xlsm'])
    
    # Pulsante per avviare il fine-tuning
    col1, col2 = st.columns(2)
    with col1:
        fine_tune_button = st.button("Avvia Fine-Tuning", type="primary", use_container_width=True, disabled=(st.session_state.model_state == 'training'))
    with col2:
        delete_corpus_button = st.button("Elimina Corpus", type="secondary", use_container_width=True, disabled=(st.session_state.model_state == 'training'))

    if fine_tune_button:
        fine_tune_model_flow(fine_tune_file_input)
    if delete_corpus_button:
        cb.delete_corpus(update_status)
        st.session_state.model_exists = False
        st.session_state.corpus_exists = False
        st.rerun()

st.markdown("---")
st.header("2. Generazione di Giudizi per File Excel")

with st.container(border=True):
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna e potrai scaricare il file aggiornato.")
    excel_file_input = st.file_uploader("Carica file Excel da completare", type=['xlsx', 'xls', 'xlsm'], key="excel_gen_uploader")
    
    # Logica per gestire il cambio di file e l'elenco dei fogli
    excel_sheet_dropdown_options = []
    process_excel_button_disabled = True
    
    if excel_file_input:
        try:
            st.session_state.excel_file_path = BytesIO(excel_file_input.getvalue())
            sheet_names = er.get_excel_sheet_names(st.session_state.excel_file_path, update_status)
            excel_sheet_dropdown_options = sheet_names
            process_excel_button_disabled = not st.session_state.model_exists or not sheet_names or (st.session_state.model_state == 'generating')
        except Exception as e:
            update_status(f"Errore nella lettura dei fogli del file: {e}", "error")
            st.session_state.excel_file_path = None

    selected_sheet = st.selectbox("Seleziona Foglio di Lavoro", options=excel_sheet_dropdown_options, index=None, placeholder="Seleziona un foglio...")
    
    # Se un foglio è stato selezionato, lo memorizziamo
    if selected_sheet:
        st.session_state.selected_sheet = selected_sheet
    else:
        st.session_state.selected_sheet = None

    if st.button("Avvia Generazione su File", type="primary", use_container_width=True, disabled=process_excel_button_disabled or not st.session_state.selected_sheet):
        process_excel_for_judgments()

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
    elif "Successo" in message:
        st.success(message)
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
        file_name=f"Giudizi_Generati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
