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
import excel_reader_v2 as er # IMPORTAZIONE CORRETTA
import model_trainer as mt
import judgment_generator as jg
import corpus_builder as cb
from config import OUTPUT_DIR, CORPUS_FILE, MODEL_NAME

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

# Funzione per registrare i messaggi di stato
def progress_container(placeholder, message, type="info"):
    """
    Mostra un messaggio di stato in un contenitore specifico.
    
    Args:
        placeholder (streamlit placeholder): Il contenitore dove mostrare il messaggio.
        message (str): Il testo del messaggio.
        type (str): Il tipo di messaggio ('info', 'success', 'warning', 'error').
    """
    with placeholder:
        if type == "info":
            st.info(message)
        elif type == "success":
            st.success(message)
        elif type == "warning":
            st.warning(message)
        elif type == "error":
            st.error(message)
        time.sleep(0.5)

# Funzione per resettare lo stato del progetto
def reset_project_state():
    """
    Elimina i file del modello e del corpus e resetta le session_state.
    """
    # Elimina il modello fine-tuned
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    # Elimina il corpus di addestramento
    if os.path.exists(CORPUS_FILE):
        os.remove(CORPUS_FILE)

    # Resetta le session_state
    for key in list(st.session_state.keys()):
        # Mantieni solo le chiavi essenziali per l'inizializzazione, se necessario
        if key not in ['corpus_df', 'model_ready', 'process_completed_file', 'uploaded_training_file', 'uploaded_process_file']:
            del st.session_state[key]
    
    # Reimposta esplicitamente le variabili principali
    st.session_state.corpus_df = pd.DataFrame()
    st.session_state.model_ready = False
    st.session_state.process_completed_file = None
    st.session_state.uploaded_training_file = None
    st.session_state.uploaded_process_file = None
    
# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE E LOGICA APPLICATIVA
# ==============================================================================

# Configurazione della pagina e titoli principali
st.set_page_config(
    page_title="Giudizi AI",
    page_icon="🤖",
    layout="wide",
)

# Inizializzazione della session_state per tutte le variabili necessarie
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'uploaded_training_file' not in st.session_state:
    st.session_state.uploaded_training_file = None
if 'uploaded_process_file' not in st.session_state:
    st.session_state.uploaded_process_file = None

st.title("🤖 Giudizi AI")
st.markdown("---")

# 1. Addestramento del Modello
st.header("1. Addestramento del Modello")
st.write("Carica qui i file di valutazione che contengono i giudizi che il modello deve imparare. Se il corpus esiste già, i nuovi dati verranno aggiunti.")

# Creazione delle colonne per organizzare gli elementi
col1, col2, col3 = st.columns(3)

with col1:
    # Creazione del selettore di file per i file di addestramento
    # Salviamo il file caricato nella session_state per non perderlo
    st.session_state.uploaded_training_file = st.file_uploader(
        "Carica file di addestramento",
        type=['xlsx', 'xls', 'xlsm'],
        key="training_uploader"
    )

with col2:
    # Creazione del pulsante per caricare e aggiornare il corpus
    if st.button("Carica e Aggiorna Corpus"):
        if st.session_state.uploaded_training_file is not None:
            status_placeholder_corpus = st.empty()
            progress_container(status_placeholder_corpus, "Avvio del caricamento e aggiornamento del corpus...", "info")
            try:
                with st.spinner("Elaborazione del file..."):
                    df_from_excel = er.read_and_prepare_data_from_excel(st.session_state.uploaded_training_file, progress_container=lambda ph, msg, type: progress_container(status_placeholder_corpus, msg, type))
                
                st.session_state.corpus_df = cb.build_or_update_corpus(df_from_excel, lambda ph, msg, type: progress_container(status_placeholder_corpus, msg, type))
                progress_container(status_placeholder_corpus, "Corpus aggiornato con successo!", "success")
                st.session_state.model_ready = False
                
            except Exception as e:
                progress_container(status_placeholder_corpus, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
        else:
            st.warning("Per favore, carica un file prima di aggiornare il corpus.")

with col3:
    # Creazione del pulsante per avviare l'addestramento del modello
    if st.button("Addestra/Aggiorna Modello"):
        if not st.session_state.corpus_df.empty:
            status_placeholder_train = st.empty()
            with st.spinner("Addestramento del modello in corso... Potrebbe richiedere diversi minuti."):
                model, tokenizer = mt.train_model(st.session_state.corpus_df, lambda ph, msg, type: progress_container(status_placeholder_train, msg, type))
            if model and tokenizer:
                st.session_state.model_ready = True
                progress_container(status_placeholder_train, "Addestramento completato. Il modello è pronto per generare giudizi.", "success")
            else:
                st.session_state.model_ready = False
                progress_container(status_placeholder_train, "Addestramento fallito. Controlla la console per i dettagli.", "error")
        else:
            st.warning("Il corpus di addestramento è vuoto. Carica prima dei dati.")

st.markdown("---")

# 2. Generazione dei Giudizi
st.header("2. Generazione dei Giudizi")

if st.session_state.model_ready:
    st.write("Ora puoi caricare un file senza giudizi e usare il modello addestrato per generarli.")
    # Creazione del selettore di file per i file da processare
    # Salviamo il file caricato nella session_state per non perderlo
    st.session_state.uploaded_process_file = st.file_uploader(
        "Carica file da processare",
        type=['xlsx', 'xls', 'xlsm'],
        key="processing_uploader"
    )

    if st.session_state.uploaded_process_file:
        sheet_names = er.get_excel_sheet_names(st.session_state.uploaded_process_file)
        if sheet_names:
            # Creazione del selettore per i fogli di lavoro
            st.session_state.selected_sheet = st.selectbox("Seleziona il foglio da processare:", sheet_names, key="sheet_selector")
        
        # Creazione del pulsante per avviare la generazione dei giudizi
        if st.button("Genera Giudizi"):
            if st.session_state.uploaded_process_file is not None and st.session_state.selected_sheet is not None:
                status_placeholder_generate = st.empty()
                progress_container(status_placeholder_generate, "Avvio della generazione dei giudizi...", "info")
                try:
                    df_to_process = er.read_and_prepare_data_from_excel(st.session_state.uploaded_process_file, sheets_to_read=[st.session_state.selected_sheet], progress_container=lambda ph, msg, type: progress_container(status_placeholder_generate, msg, type))
                    if not df_to_process.empty:
                        with st.spinner("Generazione dei giudizi in corso..."):
                            st.session_state.process_completed_file = jg.generate_judgments(df_to_process, st.session_state.selected_sheet, lambda ph, msg, type: progress_container(status_placeholder_generate, msg, type))
                        progress_container(status_placeholder_generate, "Giudizi generati con successo!", "success")
                    else:
                        progress_container(status_placeholder_generate, "Il file da processare è vuoto o non contiene dati validi.", "error")
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
    
    # Creazione del pulsante di download per il file Excel
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

st.write("### Gestione Progetto")
# Creazione di colonne per il pulsante di reset e lo stato
col_reset, col_status = st.columns([1, 2])

with col_reset:
    # Creazione del pulsante per resettare il progetto
    if st.button("Resetta tutto il progetto"):
        reset_project_state()
        st.session_state.model_ready = False
        st.rerun()

with col_status:
    # Mostra i messaggi di stato del progetto
    if st.session_state.model_ready:
        st.success("Modello pronto per la generazione di giudizi!")
    else:
        st.warning("Nessun modello addestrato. Per favore, carica un file e addestra il modello.")
    
    if not st.session_state.corpus_df.empty:
        st.info(f"Corpus di addestramento: {len(st.session_state.corpus_df)} righe")
    else:
        st.info("Corpus di addestramento: vuoto")
