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
import requests
import re

# Importa i moduli personalizzati
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, CORPUS_FILE, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Definizione della variabile dell'API Key per l'uso con le API di Gemini
GEMINI_API_KEY = "" # Sostituisci con la tua chiave API se usi gemini

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def progress_container(placeholder, message, type):
    """
    Gestisce l'aggiornamento del placeholder di stato con un messaggio.
    """
    if placeholder:
        if type == "info":
            placeholder.info(message)
        elif type == "success":
            placeholder.success(message)
        elif type == "warning":
            placeholder.warning(message)
        elif type == "error":
            placeholder.error(message)

def clear_cache():
    """
    Funzione per eliminare la cache di Streamlit.
    """
    st.legacy_caching.clear_cache()
    progress_container(None, "Cache di Streamlit pulita.", "info")

def reset_project_state():
    """
    Resetta tutti i file del progetto (corpus, modello).
    """
    # Elimina il corpus
    cb.delete_corpus(lambda msg, type: progress_container(None, msg, type))
    # Elimina il modello
    mt.delete_model(lambda msg, type: progress_container(None, msg, type))
    progress_container(None, "Progetto resettato. Tutti i dati sono stati eliminati.", "success")
    # Reset del session state per i file caricati
    if 'uploaded_files' in st.session_state:
        del st.session_state['uploaded_files']
    if 'uploaded_process_file' in st.session_state:
        del st.session_state['uploaded_process_file']

def get_session_id():
    """
    Genera un ID di sessione per distinguere i file in un ambiente multi-utente.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    return st.session_state.session_id

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE E LOGICA PRINCIPALE
# ==============================================================================

# Impostazioni della pagina
st.set_page_config(
    page_title="Giudizi-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializzazione di Streamlit session state per mantenere lo stato dell'app
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'selected_sheet_trainer' not in st.session_state:
    st.session_state.selected_sheet_trainer = None
if 'uploaded_process_file' not in st.session_state:
    st.session_state.uploaded_process_file = None
if 'selected_sheet_process' not in st.session_state:
    st.session_state.selected_sheet_process = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None

# Titolo e descrizione dell'app
st.title("🤖 Giudizi-AI: Generatore di Giudizi 🤖")
st.markdown("---")
st.markdown("Questa applicazione ti aiuta a generare automaticamente giudizi per le tue valutazioni. Segui i passaggi di seguito.")

# Layout in colonne per la sidebar e il contenuto principale
main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    st.markdown("### Impostazioni e Stato")
    
    st.info("Utilizza questa sidebar per resettare l'applicazione o per informazioni di stato.")
    
    # Bottone per caricare il modello
    if st.button("Carica Modello Esistente"):
        status_placeholder_sidebar = st.empty()
        model, tokenizer = mt.load_fine_tuned_model(lambda msg, type: progress_container(status_placeholder_sidebar, msg, type))
        if model and tokenizer:
            st.session_state.model_ready = True
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            progress_container(status_placeholder_sidebar, "Modello caricato con successo.", "success")
        else:
            st.session_state.model_ready = False
            progress_container(status_placeholder_sidebar, "Nessun modello trovato. Devi addestrarne uno.", "error")
            
    # Bottone per resettare lo stato del progetto
    if st.button("Resetta tutto il progetto"):
        reset_project_state()
        st.session_state.model_ready = False
        st.experimental_rerun()
        
    st.markdown("---")
    st.write("### Stato Attuale")
    if st.session_state.model_ready:
        st.success("✅ Modello caricato e pronto!")
    else:
        st.error("❌ Nessun modello caricato.")
        
    st.write(f"Righe nel corpus: {len(st.session_state.corpus_df)}")
    
    # Mostra l'ID della sessione
    st.text(f"ID Sessione: {get_session_id()}")

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
with main_col:
    st.header("1. Addestramento del Modello 🧠")
    st.info("Carica uno o più file Excel per creare o aggiornare il corpus di addestramento. Le colonne 'Descrizione' e 'Giudizio' verranno utilizzate per il fine-tuning del modello.")
    
    # Placeholder per i messaggi di stato
    status_placeholder_trainer = st.empty()
    
    # Caricamento dei file per l'addestramento
    uploaded_files_trainer = st.file_uploader(
        "Carica i file di addestramento (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="uploader_trainer"
    )
    
    if uploaded_files_trainer:
        st.session_state.uploaded_files = uploaded_files_trainer
        # Mostra i fogli di lavoro e permette di selezionarne uno
        first_file = uploaded_files_trainer[0]
        sheet_names = er.get_excel_sheet_names(first_file)
        st.session_state.selected_sheet_trainer = st.selectbox(
            "Seleziona il foglio di lavoro da utilizzare per l'addestramento:",
            sheet_names,
            key="sheet_select_trainer"
        )
        
    if st.button("Avvia Addestramento"):
        if st.session_state.uploaded_files and st.session_state.selected_sheet_trainer:
            try:
                # Legge e prepara il nuovo dataframe
                status_placeholder_trainer.info("Lettura dei file e preparazione del corpus...")
                new_dfs = [er.read_excel_file(f, st.session_state.selected_sheet_trainer, lambda msg, type: progress_container(status_placeholder_trainer, msg, type)) for f in st.session_state.uploaded_files]
                new_df = pd.concat(new_dfs).reset_index(drop=True)

                # Aggiorna il corpus esistente
                st.session_state.corpus_df = cb.build_or_update_corpus(new_df, lambda msg, type: progress_container(status_placeholder_trainer, msg, type))
                
                # Avvia il fine-tuning se il corpus non è vuoto
                if not st.session_state.corpus_df.empty:
                    st.session_state.model, st.session_state.tokenizer = mt.train_model(st.session_state.corpus_df, lambda msg, type: progress_container(status_placeholder_trainer, msg, type))
                    if st.session_state.model and st.session_state.tokenizer:
                        st.session_state.model_ready = True
                        progress_container(status_placeholder_trainer, "Addestramento completato con successo!", "success")
                        st.balloons()
                    else:
                        st.session_state.model_ready = False
                        progress_container(status_placeholder_trainer, "Errore durante l'addestramento. Controlla i dati.", "error")
                else:
                    progress_container(status_placeholder_trainer, "Corpus vuoto. Impossibile avviare l'addestramento.", "error")
            except Exception as e:
                progress_container(status_placeholder_trainer, f"Errore durante il processo di addestramento: {e}", "error")
                st.error(f"Errore: {traceback.format_exc()}")
        else:
            st.warning("Per addestrare il modello, devi prima caricare almeno un file e selezionare un foglio.")

st.markdown("---")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
with main_col:
    st.header("2. Generazione dei Giudizi 🤖")
    st.info("Carica il file Excel che vuoi completare. L'applicazione genererà i giudizi per le righe mancanti.")
    
    # Placeholder per i messaggi di stato
    status_placeholder_generate = st.empty()

    # Caricamento del file per la generazione
    uploaded_process_file = st.file_uploader(
        "Carica il file da completare (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="uploader_process"
    )
    
    if uploaded_process_file:
        st.session_state.uploaded_process_file = uploaded_process_file
        # Mostra i fogli di lavoro e permette di selezionarne uno
        sheet_names = er.get_excel_sheet_names(uploaded_process_file)
        st.session_state.selected_sheet_process = st.selectbox(
            "Seleziona il foglio di lavoro da completare:",
            sheet_names,
            key="sheet_select_process"
        )
        
    if st.button("Genera Giudizi"):
        if st.session_state.model_ready:
            if st.session_state.uploaded_process_file and st.session_state.selected_sheet_process:
                try:
                    progress_container(status_placeholder_generate, "Lettura del file...", "info")
                    df = er.read_excel_file(st.session_state.uploaded_process_file, st.session_state.selected_sheet_process, lambda msg, type: progress_container(status_placeholder_generate, msg, type))
                    
                    if not df.empty:
                        progress_container(status_placeholder_generate, "Generazione dei giudizi in corso...", "info")
                        # Usa il modello e il tokenizer dalla sessione
                        df_with_judgments = jg.generate_judgments(df, st.session_state.model, st.session_state.tokenizer, st.session_state.selected_sheet_process, lambda msg, type: progress_container(status_placeholder_generate, msg, type))
                        
                        st.session_state.process_completed_file = df_with_judgments
                        st.session_state.selected_sheet = st.session_state.selected_sheet_process
                        progress_container(status_placeholder_generate, "Generazione completata!", "success")
                    else:
                        progress_container(status_placeholder_generate, "Il file caricato è vuoto o non contiene dati validi.", "error")
                except Exception as e:
                    progress_container(status_placeholder_generate, f"Errore nella generazione dei giudizi: {e}", "error")
                    st.error(f"Errore: {traceback.format_exc()}")
            else:
                st.warning("Per generare i giudizi, devi prima caricare un file nella sezione '2. Generazione dei Giudizi'.")
        else:
            st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

st.markdown("---")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
with main_col:
    st.header("3. Stato e Download")

    if st.session_state.process_completed_file is not None:
        st.write("### Scarica il file completato")
        st.write("Di seguito, puoi vedere l'anteprima del file con i giudizi generati e scaricare il documento completo.")
        
        st.dataframe(st.session_state.process_completed_file)
        
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
    else:
        st.info("I risultati appariranno qui una volta che avrai generato i giudizi.")
