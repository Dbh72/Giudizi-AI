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

# Inizializziamo lo stato della sessione per mantenere le variabili tra i run
if 'corpus_loaded' not in st.session_state:
    st.session_state.corpus_loaded = False
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def progress_container(message, message_type="info"):
    """
    Funzione per mostrare messaggi di stato all'utente e registrarli nel log.
    I messaggi persistono nello stato della sessione.
    """
    st.session_state.log_messages.append({
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'text': message,
        'type': message_type
    })
    # Mostra i messaggi più recenti nell'interfaccia
    if message_type == "info":
        st.info(message)
    elif message_type == "success":
        st.success(message)
    elif message_type == "warning":
        st.warning(message)
    elif message_type == "error":
        st.error(message)

# ==============================================================================
# INTERFACCIA UTENTE CON STREAMLIT
# ==============================================================================

# Titolo e sottotitolo dell'applicazione
st.title("🤖 Giudizi-AI: Il Tuo Assistente per le Valutazioni")
st.markdown("Crea un modello personalizzato per generare giudizi automaticamente per le verifiche.")

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.markdown("---")
st.header("1. Addestramento del Modello")

st.write("### Carica i dati di addestramento")
st.write("Carica uno o più file Excel (`.xlsx`, `.xls`, `.xlsm`) contenenti le tue valutazioni per addestrare il modello.")
uploaded_train_file = st.file_uploader("Carica file di addestramento", type=["xlsx", "xls", "xlsm"])

# Bottone per aggiornare il corpus
if st.button("Aggiorna Corpus"):
    if uploaded_train_file:
        progress_container("Lettura del file Excel e preparazione del corpus...", "info")
        # Legge il file e lo prepara in un DataFrame
        corpus_df = er.read_and_prepare_data_from_excel(uploaded_train_file, progress_container)
        # Aggiorna il corpus esistente con il nuovo DataFrame
        updated_corpus_df = cb.build_or_update_corpus(corpus_df, progress_container)
        
        if not updated_corpus_df.empty:
            progress_container(f"Corpus aggiornato con successo! Righe totali: {len(updated_corpus_df)}", "success")
            st.session_state.corpus_loaded = True
        else:
            progress_container("Aggiornamento del corpus fallito. Controlla il file e riprova.", "error")
    else:
        progress_container("Devi caricare un file per aggiornare il corpus.", "warning")

st.markdown("---")
st.write("### Addestra/Aggiorna il Modello")
st.write("Una volta che il corpus è pronto, clicca qui per addestrare il modello.")

# Bottone per avviare l'addestramento
if st.button("Addestra/Aggiorna Modello"):
    if st.session_state.corpus_loaded or os.path.exists(CORPUS_FILE):
        progress_container("Caricamento del corpus di addestramento...", "info")
        corpus_df = cb.load_corpus(progress_container)
        
        if not corpus_df.empty:
            progress_container("Avvio dell'addestramento del modello...", "info")
            st.session_state.model, st.session_state.tokenizer = mt.train_model(corpus_df, progress_container)
            
            if st.session_state.model and st.session_state.tokenizer:
                progress_container("Modello addestrato con successo e pronto per l'uso!", "success")
                st.session_state.model_loaded = True
            else:
                progress_container("Addestramento fallito. Controlla il log per i dettagli.", "error")
                
        else:
        	progress_container("Nessun corpus valido trovato. Carica un file per creare il corpus e riprova l'addestramento.", "warning")
            
    else:
        progress_container("Per addestrare il modello, devi prima aggiornare il corpus.", "warning")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.markdown("---")
st.header("2. Generazione dei Giudizi")

st.write("### Carica il file da completare")
st.write("Carica il file Excel (`.xlsx`, `.xls`, `.xlsm`) di cui vuoi generare i giudizi. Assicurati che il file contenga la colonna 'Giudizio'.")
uploaded_process_file = st.file_uploader("Carica file da completare", type=["xlsx", "xls", "xlsm"])

if uploaded_process_file:
    # Get sheet names and let the user select one
    sheet_names = er.get_excel_sheet_names(uploaded_process_file)
    st.session_state.selected_sheet = st.selectbox("Seleziona il foglio di lavoro", sheet_names, key="sheet_selector")
    
    # Bottone per generare i giudizi
    if st.button("Genera Giudizi"):
        if st.session_state.model_loaded:
            if uploaded_process_file and st.session_state.selected_sheet:
                progress_container("Generazione dei giudizi in corso...", "info")
                try:
                    df = er.read_single_sheet(uploaded_process_file, st.session_state.selected_sheet, progress_container)
                    st.session_state.process_completed_file = jg.generate_judgments(df, st.session_state.model, st.session_state.tokenizer, st.session_state.selected_sheet, progress_container)
                    
                    if st.session_state.process_completed_file is not None:
                        progress_container("Generazione dei giudizi completata con successo!", "success")
                    else:
                        progress_container("Generazione dei giudizi fallita. Controlla il log per i dettagli.", "error")
                except Exception as e:
                    progress_container(f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
            else:
                progress_container("Per generare i giudizi, devi prima caricare un file e selezionare un foglio.", "warning")
        else:
            progress_container("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.", "warning")

# ==============================================================================
# SEZIONE 3: STATO E DOWNLOAD
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

# Sezione per il log degli eventi
st.markdown("---")
st.header("4. Log degli Eventi")
st.write("Tutti i messaggi di stato vengono registrati qui.")

with st.expander("Mostra Log"):
    log_area = st.container()
    for entry in st.session_state.log_messages:
        text = f"[{entry['timestamp']}] {entry['text']}"
        if entry['type'] == 'info':
            log_area.info(text)
        elif entry['type'] == 'success':
            log_area.success(text)
        elif entry['type'] == 'warning':
            log_area.warning(text)
        elif entry['type'] == 'error':
            log_area.error(text)
            
    if st.button("Cancella Log"):
        st.session_state.log_messages = []
        st.experimental_rerun()
