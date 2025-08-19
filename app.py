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
import sys

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

def log_and_display_message(message, type="info"):
    """
    Funzione per stampare un messaggio in console e aggiungerlo alla session_state di Streamlit.
    """
    if 'status_messages' not in st.session_state:
        st.session_state.status_messages = []
    
    print(f"[{type.upper()}] {message}")
    st.session_state.status_messages.append({'message': message, 'type': type})

def clear_session_state():
    """
    Pulisce lo stato della sessione per riavviare l'applicazione.
    """
    st.session_state.status_messages = []
    st.session_state.fine_tuning_state = 'initial'
    st.session_state.model_state = 'not_loaded'
    st.session_state.trained_model = None
    st.session_state.tokenizer = None
    st.session_state.process_completed_file = None
    st.session_state.selected_sheet = None
    st.session_state.training_df = None
    st.session_state.file_uploaded_for_training = False
    
# Inizializzazione dello stato della sessione
if 'status_messages' not in st.session_state:
    clear_session_state()

# ==============================================================================
# SEZIONE 2: LAYOUT E LOGICA DELL'INTERFACCIA UTENTE
# ==============================================================================

st.title("📚 Generatore di Giudizi per la Scuola AI")
st.markdown("---")
st.markdown(
    """
    Benvenuto nel Generatore di Giudizi AI!
    Questa applicazione ti permette di:
    1.  **Addestrare (Fine-Tuning) un modello**: Carica un file Excel contenente il corpus di addestramento.
        Il modello imparerà a generare giudizi basati sui tuoi dati.
        Il modello addestrato e il corpus di dati verranno salvati in locale per riutilizzi futuri.
    2.  **Generare Giudizi**: Carica un file Excel con la colonna 'Giudizio' vuota.
        Il modello completerà la colonna e potrai scaricare il file aggiornato.
    """
)
st.markdown("---")

# Rilevamento automatico e caricamento del modello
log_and_display_message("Avvio dell'applicazione...", "info")
if st.session_state.model_state == 'not_loaded':
    log_and_display_message("Controllo della presenza di un modello fine-tuned...", "info")
    if os.path.exists(OUTPUT_DIR):
        log_and_display_message(f"Trovata directory del modello '{OUTPUT_DIR}'. Caricamento in corso...", "info")
        st.session_state.trained_model, st.session_state.tokenizer = jg.load_trained_model(OUTPUT_DIR, log_and_display_message)
        if st.session_state.trained_model and st.session_state.tokenizer:
            st.session_state.model_state = 'ready'
            log_and_display_message("Modello fine-tuned caricato con successo!", "success")
        else:
            st.session_state.model_state = 'failed'
            log_and_display_message("Errore nel caricamento del modello fine-tuned.", "error")
    else:
        log_and_display_message("Nessun modello fine-tuned trovato. Sarà possibile addestrarne uno nuovo.", "info")
        st.session_state.model_state = 'untrained'
        
st.markdown("---")
st.header("1. Addestramento del Modello")
st.markdown(
    """
    **Addestra il modello** caricando un file Excel con i dati di esempio.
    Assicurati che il file contenga una colonna 'Giudizio' e le altre colonne
    necessarie per la descrizione dello studente.
    """
)

# Aggiungi un'opzione per eliminare il corpus di addestramento
if os.path.exists(CORPUS_FILE):
    st.info("Corpus di addestramento esistente. Puoi caricare un nuovo file per aggiornarlo o eliminarlo.")
    if st.button("Elimina Corpus di Addestramento"):
        cb.delete_corpus(log_and_display_message)
        st.rerun()

fine_tune_file = st.file_uploader(
    "Carica qui il tuo file Excel per il fine-tuning:",
    type=["xlsx", "xls", "xlsm"],
    key="fine_tune_file_uploader",
    help="Carica un file Excel con i dati per l'addestramento del modello."
)

if fine_tune_file:
    log_and_display_message(f"File '{fine_tune_file.name}' caricato per l'addestramento.", "info")
    st.session_state.file_uploaded_for_training = True
    st.session_state.training_df = er.read_excel_file_to_df(BytesIO(fine_tune_file.getvalue()), log_and_display_message)

    if not st.session_state.training_df.empty:
        st.success(f"Dati di addestramento letti con successo. Trovate {len(st.session_state.training_df)} righe.")
        if st.button("Avvia Fine-Tuning"):
            if st.session_state.fine_tuning_state == 'initial':
                try:
                    log_and_display_message("Avvio del processo di fine-tuning...", "info")
                    st.session_state.fine_tuning_state = 'running'
                    
                    # Aggiorna il corpus di dati
                    corpus_df = cb.build_or_update_corpus(st.session_state.training_df, log_and_display_message)

                    if not corpus_df.empty:
                        # Avvia il fine-tuning
                        st.session_state.trained_model, st.session_state.tokenizer = mt.fine_tune(corpus_df, log_and_display_message)
                        
                        if st.session_state.trained_model and st.session_state.tokenizer:
                            log_and_display_message("Fine-tuning completato con successo!", "success")
                            st.session_state.model_state = 'ready'
                        else:
                            log_and_display_message("Fine-tuning fallito.", "error")
                            st.session_state.model_state = 'failed'
                            
                    else:
                        log_and_display_message("Impossibile procedere: il corpus di addestramento è vuoto.", "error")
                        st.session_state.model_state = 'failed'
                        
                except Exception as e:
                    log_and_display_message(f"Errore critico durante il fine-tuning: {e}", "error")
                    log_and_display_message(f"Traceback: {traceback.format_exc()}", "error")
                    st.session_state.fine_tuning_state = 'error'
                    st.session_state.model_state = 'failed'
                finally:
                    st.session_state.fine_tuning_state = 'initial'
                    st.rerun()
            else:
                log_and_display_message("Il fine-tuning è già in esecuzione. Attendi il completamento.", "warning")
    else:
        st.error("Il file Excel non contiene dati validi per l'addestramento.")

st.markdown("---")
st.header("2. Generazione di Giudizi")

if st.session_state.model_state == 'ready':
    st.success("Modello pronto per la generazione di giudizi!")
    
    excel_file_to_complete = st.file_uploader(
        "Carica qui il file Excel da completare:",
        type=["xlsx", "xls", "xlsm"],
        key="excel_to_complete_uploader",
        help="Carica un file Excel con la colonna 'Giudizio' da riempire."
    )
    
    if excel_file_to_complete:
        st.info("File caricato. Seleziona il foglio di lavoro.")
        
        try:
            # Leggi i nomi dei fogli
            excel_file_bytes = BytesIO(excel_file_to_complete.getvalue())
            xl = pd.ExcelFile(excel_file_bytes)
            sheet_names = xl.sheet_names
            
            selected_sheet = st.selectbox("Seleziona il Foglio di Lavoro:", sheet_names, key="selected_sheet")
            
            if selected_sheet:
                st.session_state.selected_sheet = selected_sheet
                
                if st.button("Avvia Generazione"):
                    try:
                        log_and_display_message("Lettura del file Excel da completare...", "info")
                        df_to_complete = pd.read_excel(excel_file_bytes, sheet_name=selected_sheet)
                        
                        giudizio_col = er.find_giudizio_column(df_to_complete)
                        
                        if giudizio_col:
                            log_and_display_message(f"Colonna 'Giudizio' trovata: '{giudizio_col}'. Avvio della generazione...", "info")
                            
                            st.session_state.process_completed_file = jg.generate_judgments_for_excel(
                                st.session_state.trained_model, 
                                st.session_state.tokenizer, 
                                df_to_complete, 
                                giudizio_col, 
                                selected_sheet, 
                                OUTPUT_DIR, 
                                log_and_display_message
                            )
                            
                            st.success("Generazione completata con successo!")
                            st.balloons()
                            
                        else:
                            log_and_display_message("Colonna 'Giudizio' non trovata nel foglio selezionato.", "error")
                            st.error("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che l'intestazione esista.")
                    except Exception as e:
                        log_and_display_message(f"Errore nella lettura del file Excel: {e}", "error")
                        log_and_display_message(f"Traceback: {traceback.format_exc()}", "error")
                        
        except Exception as e:
            log_and_display_message(f"Errore nel caricamento del file: {e}", "error")
            log_and_display_message(f"Traceback: {traceback.format_exc()}", "error")
            st.error("Errore nel caricamento del file. Controlla il formato e riprova.")

else:
    st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.header("3. Stato e Download")

# Visualizza i messaggi di stato
for message in st.session_state.status_messages:
    if message['type'] == "error":
        st.error(message['message'])
    elif message['type'] == "warning":
        st.warning(message['message'])
    else:
        st.info(message['message'])
        
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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

