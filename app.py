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
import zipfile

# Importa i moduli personalizzati
import excel_reader as er
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, CORPUS_FILE, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Definiamo la funzione per i messaggi di progresso e logging
def progress_container(placeholder, message, type="info"):
    """
    Mostra un messaggio di progresso nell'interfaccia utente e lo salva in un file di log.
    
    Args:
        placeholder: Il placeholder di Streamlit in cui mostrare il messaggio.
        message (str): Il testo del messaggio da visualizzare.
        type (str): Il tipo di messaggio ("info", "success", "error", "warning").
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] [{type.upper()}] {message}\n"
    
    with open(os.path.join(OUTPUT_DIR, "Logs.txt"), "a") as f:
        f.write(log_message)

    if type == "info":
        placeholder.info(message)
    elif type == "success":
        placeholder.success(message)
    elif type == "error":
        placeholder.error(message)
    elif type == "warning":
        placeholder.warning(message)

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

# Funzione per registrare i progressi
@st.cache_resource
def setup_directories():
    """Imposta le directory di output se non esistono."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Inizializza o resetta il file di log all'avvio dell'app
    log_path = os.path.join(OUTPUT_DIR, "Logs.txt")
    with open(log_path, "w") as f:
        f.write(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        
def create_model_zip(model_path, zip_path):
    """
    Crea un file ZIP contenente la cartella del modello.
    
    Args:
        model_path (str): Il percorso della cartella del modello.
        zip_path (str): Il percorso del file ZIP da creare.
    """
    shutil.make_archive(zip_path.replace('.zip', ''), 'zip', model_path)
    return zip_path

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE PRINCIPALE (STREAMLIT)
# ==============================================================================
st.title("Generatore di Giudizi per Aziende")
st.markdown("### Modulo di Addestramento e Generazione Giudizi")

setup_directories()

# Initialize session state variables
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'trained_tokenizer' not in st.session_state:
    st.session_state.trained_tokenizer = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# Mappa la funzione di logging per renderla disponibile in altri moduli
st.session_state['progress_container'] = progress_container

# ==============================================================================
# SEZIONE 2.1: Addestramento del Modello
# ==============================================================================
st.header("1. Addestramento del Modello")

uploaded_train_file = st.file_uploader("Carica il file Excel per l'addestramento (opzionale)", type=["xlsx"])
if uploaded_train_file:
    # Mostra i nomi dei fogli di lavoro e permette la selezione
    excel_sheets = pd.ExcelFile(uploaded_train_file).sheet_names
    selected_sheet_train = st.selectbox("Seleziona il foglio di lavoro per l'addestramento", excel_sheets)

    # Pulsante per avviare il processo di addestramento
    if st.button("Avvia Addestramento"):
        with st.spinner('Addestramento in corso...'):
            status_placeholder_train = st.empty()
            
            try:
                # Step 1: Costruzione o aggiornamento del Corpus
                progress_container(status_placeholder_train, "Costruzione del corpus iniziale...", "info")
                df_train = er.load_excel_to_df(uploaded_train_file, selected_sheet_train)
                corpus_path = os.path.join(OUTPUT_DIR, CORPUS_FILE)
                
                # Questa funzione aggiornerà il corpus con il nuovo file
                cb.build_corpus(df_train, corpus_path)
                
                progress_container(status_placeholder_train, "Corpus aggiornato. Avvio dell'addestramento...", "info")

                # Step 2: Addestramento del modello
                st.session_state.trained_model, st.session_state.trained_tokenizer = mt.fine_tune_model(corpus_path, status_placeholder_train)
                
                progress_container(status_placeholder_train, "Addestramento completato con successo!", "success")
                st.balloons()
            except Exception as e:
                progress_container(status_placeholder_train, f"Errore critico durante l'addestramento: {e}", "error")
                progress_container(status_placeholder_train, f"Traceback: {traceback.format_exc()}", "error")

# ==============================================================================
# SEZIONE 2.2: Generazione Giudizi
# ==============================================================================
st.markdown("---")
st.header("2. Generazione dei Giudizi")

if st.session_state.trained_model is not None:
    uploaded_process_file = st.file_uploader("Carica il file Excel da completare", type=["xlsx"])
    if uploaded_process_file:
        excel_sheets = pd.ExcelFile(uploaded_process_file).sheet_names
        st.session_state.selected_sheet = st.selectbox("Seleziona il foglio di lavoro da processare", excel_sheets)
        
        if st.button("Genera Giudizi"):
            with st.spinner('Generazione dei giudizi...'):
                status_placeholder_generate = st.empty()
                
                try:
                    df_process = er.load_excel_to_df(uploaded_process_file, st.session_state.selected_sheet)
                    st.session_state.process_completed_file = jg.generate_judgments(df_process, st.session_state.trained_model, st.session_state.trained_tokenizer, status_placeholder_generate)
                    progress_container(status_placeholder_generate, "Generazione dei giudizi completata!", "success")
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

# Aggiunge i pulsanti per il download del modello e dei log
st.write("### Scarica il Modello e i Log")
if st.session_state.trained_model is not None:
    # Crea un file zip del modello per il download
    model_zip_path = os.path.join(OUTPUT_DIR, "final_model.zip")
    if os.path.exists(os.path.join(OUTPUT_DIR, "final_model")):
        create_model_zip(os.path.join(OUTPUT_DIR, "final_model"), model_zip_path)
    
        with open(model_zip_path, "rb") as fp:
            st.download_button(
                label="Scarica Modello Finale (ZIP)",
                data=fp,
                file_name="final_model.zip",
                mime="application/zip"
            )

# Download del file di log
log_path = os.path.join(OUTPUT_DIR, "Logs.txt")
if os.path.exists(log_path):
    with open(log_path, "rb") as fp:
        st.download_button(
            label="Scarica Log di Addestramento",
            data=fp,
            file_name=f"Logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
else:
    st.info("I log saranno disponibili per il download dopo l'addestramento.")
