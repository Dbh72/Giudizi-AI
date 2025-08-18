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
CORPUS_FILE = "corpus.json"
STATE_FILE = "generation_state.json"

# ==============================================================================
# SEZIONE 1: GESTIONE DELLO STATO DELLA SESSIONE
# ==============================================================================
# Inizializza le variabili di stato se non esistono
if 'df_corpus' not in st.session_state:
    st.session_state.df_corpus = None
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None
if 'is_model_loaded' not in st.session_state:
    st.session_state.is_model_loaded = False
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'fine_tuning_in_progress' not in st.session_state:
    st.session_state.fine_tuning_in_progress = False

def add_status_message(message, level="info"):
    """Aggiunge un messaggio allo stato e lo stampa in console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    st.session_state.status_messages.append(formatted_message)
    if level == "info":
        st.info(message)
    elif level == "error":
        st.error(message)

# ==============================================================================
# SEZIONE 2: FUNZIONI PER I BOTTONI DELL'INTERFACCIA
# ==============================================================================

def train_model(corpus_df, num_epochs, learning_rate, batch_size, update_corpus):
    """Gestisce il processo di fine-tuning del modello."""
    add_status_message("Avvio del processo di fine-tuning...", level="info")
    st.session_state.fine_tuning_in_progress = True
    st.session_state.trained_model = None
    st.session_state.tokenizer = None
    st.session_state.is_model_loaded = False

    try:
        # Pulisce la directory del modello se non si vuole fare l'addestramento incrementale
        if not update_corpus and os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
            add_status_message(f"Cancellata la directory del modello esistente: {OUTPUT_DIR}", level="info")
        
        # Salvataggio del corpus nel file JSON per l'addestramento incrementale
        if update_corpus:
            try:
                # Leggi il corpus esistente se presente
                if os.path.exists(CORPUS_FILE):
                    existing_df = pd.read_json(CORPUS_FILE)
                    combined_df = pd.concat([existing_df, corpus_df], ignore_index=True).drop_duplicates()
                else:
                    combined_df = corpus_df
                # Salva il nuovo corpus combinato
                combined_df.to_json(CORPUS_FILE, orient='records', lines=True)
                corpus_to_train = combined_df
                add_status_message(f"Corpus aggiornato e salvato in {CORPUS_FILE}", level="info")
            except Exception as e:
                add_status_message(f"Errore nell'aggiornamento del corpus: {e}", level="error")
                return

        final_model_path = mt.fine_tune_model(corpus_df, OUTPUT_DIR, num_epochs, learning_rate, batch_size, st.session_state.status_messages)
        st.session_state.fine_tuning_in_progress = False
        add_status_message("Addestramento completato con successo!", level="info")
        st.balloons()
    except Exception as e:
        st.session_state.fine_tuning_in_progress = False
        add_status_message(f"Errore durante l'addestramento: {e}", level="error")
        st.error(f"Errore durante l'addestramento: {e}\n\nTraceback:\n{traceback.format_exc()}")
        return

    # Caricamento del modello appena addestrato
    try:
        model, tokenizer = jg.load_trained_model(final_model_path)
        st.session_state.trained_model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.is_model_loaded = True
        add_status_message("Modello appena addestrato caricato correttamente.", level="info")
    except Exception as e:
        add_status_message(f"Errore nel caricare il modello appena addestrato: {e}", level="error")
        st.error(f"Errore nel caricare il modello appena addestrato: {e}\n\nTraceback:\n{traceback.format_exc()}")

def generate_judgments_callback(file_to_complete, giudizio_col, selected_sheet):
    """Gestisce il processo di generazione dei giudizi."""
    add_status_message("Avvio della generazione dei giudizi...", level="info")
    st.session_state.process_completed_file = None
    try:
        if st.session_state.trained_model is None or st.session_state.tokenizer is None:
            raise ValueError("Il modello non è stato caricato. Si prega di addestrare o caricare un modello.")

        completed_df = jg.generate_judgments_for_excel(
            st.session_state.trained_model,
            st.session_state.tokenizer,
            file_to_complete,
            giudizio_col,
            selected_sheet,
            OUTPUT_DIR,
            st.session_state.status_messages
        )
        st.session_state.process_completed_file = completed_df
        add_status_message("Generazione completata con successo!", level="info")
        st.balloons()
    except Exception as e:
        add_status_message(f"Errore durante la generazione: {e}", level="error")
        st.error(f"Errore durante la generazione: {e}\n\nTraceback:\n{traceback.format_exc()}")

# ==============================================================================
# SEZIONE 3: INTERFACCIA UTENTE (STREAMLIT)
# ==============================================================================

# Titolo principale
st.title("Sistema di Generazione di Giudizi per Docenti 👩‍🏫")
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.2rem;
}
</style>
""",unsafe_allow_html=True)

# Tabs per le diverse sezioni dell'app
tab_trainer, tab_generator = st.tabs(["**ADDESTRA MODELLO 🧠**", "**GENERA GIUDIZI 📝**"])

# Tab 1: Addestramento del modello
with tab_trainer:
    st.header("Addestramento del Modello (Fine-Tuning)")
    st.write("Carica il tuo file Excel contenente i dati di addestramento (coppie di testo e giudizi).")

    # Caricamento del file di addestramento
    uploaded_file_trainer = st.file_uploader(
        "Carica il file per l'addestramento",
        type=['xlsx', 'xls', 'xlsm'],
        key="trainer_uploader"
    )

    if uploaded_file_trainer:
        try:
            add_status_message("Lettura del file di addestramento...", level="info")
            # Legge tutti i fogli del file Excel
            all_sheets = pd.read_excel(uploaded_file_trainer, sheet_name=None)
            sheet_names = list(all_sheets.keys())
            
            # Utilizza il reader per creare il corpus
            df_corpus = er.read_and_prepare_excel(uploaded_file_trainer, sheet_names, st.session_state.status_messages)
            
            if not df_corpus.empty:
                st.session_state.df_corpus = df_corpus
                st.success("File di addestramento caricato e analizzato correttamente!")
                st.write("### Anteprima del Corpus di Addestramento")
                st.dataframe(df_corpus.head())
            else:
                st.error("Nessun dato valido trovato per l'addestramento.")
                st.session_state.df_corpus = None

        except Exception as e:
            st.error(f"Errore nel caricare il file di addestramento: {e}\n\nTraceback:\n{traceback.format_exc()}")
            st.session_state.df_corpus = None

    if st.session_state.df_corpus is not None and not st.session_state.df_corpus.empty:
        st.markdown("---")
        st.subheader("Opzioni di Addestramento")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_epochs = st.number_input("Numero di Epochs", min_value=1, max_value=10, value=3)
        with col2:
            learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-3, value=3e-4, format="%e")
        with col3:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=2)

        update_corpus = st.checkbox("Addestramento Incrementale (aggiungi nuovi dati al corpus esistente)", value=False)
        
        st.markdown("---")
        
        if st.button("Avvia Fine-Tuning", key="train_button", disabled=st.session_state.fine_tuning_in_progress):
            st.session_state.fine_tuning_in_progress = True
            train_model(st.session_state.df_corpus, num_epochs, learning_rate, batch_size, update_corpus)
            st.session_state.fine_tuning_in_progress = False

# Tab 2: Generazione di giudizi
with tab_generator:
    st.header("Generazione di Giudizi")
    st.write("Carica il file Excel in cui vuoi che il modello generi i giudizi. Assicurati che contenga una colonna 'Giudizio'.")
    
    # Pulsante per caricare il modello addestrato, se esiste
    if st.button("Carica Modello Addestrato", key="load_model_button", disabled=st.session_state.is_model_loaded):
        add_status_message("Tentativo di caricare il modello salvato...", level="info")
        try:
            model, tokenizer = jg.load_trained_model(OUTPUT_DIR)
            st.session_state.trained_model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.is_model_loaded = True
            add_status_message("Modello caricato con successo!", level="info")
        except Exception as e:
            add_status_message("Modello non trovato o errore nel caricamento. Si prega di addestrare un modello per primo.", level="error")
            st.error(f"Errore: {e}\n\nTraceback:\n{traceback.format_exc()}")

    # Visualizzazione dello stato del modello
    model_status_col, _ = st.columns([1, 4])
    with model_status_col:
        status_text = "Modello caricato ✅" if st.session_state.is_model_loaded else "Modello non caricato ❌"
        st.markdown(f"**Stato del modello:** {status_text}")

    # Caricamento del file per la generazione
    if st.session_state.is_model_loaded:
        uploaded_file_generator = st.file_uploader(
            "Carica il file per la generazione",
            type=['xlsx', 'xls', 'xlsm'],
            key="generator_uploader"
        )

        if uploaded_file_generator:
            try:
                # Elenco dei fogli nel file caricato
                xl = pd.ExcelFile(uploaded_file_generator)
                sheet_names = xl.sheet_names
                
                # Selezione del foglio
                st.session_state.selected_sheet = st.selectbox(
                    "Seleziona il foglio di lavoro",
                    sheet_names,
                    key="sheet_selector"
                )

                # Legge il foglio selezionato in un DataFrame
                df_to_complete = xl.parse(st.session_state.selected_sheet)
                
                # Trova la colonna "Giudizio"
                giudizio_col = er.find_giudizio_column(df_to_complete)
                
                if not giudizio_col:
                    st.warning("La colonna 'Giudizio' non è stata trovata. La generazione non può proseguire.")
                else:
                    st.write("### Anteprima del File da Completare")
                    st.dataframe(df_to_complete.head())
                    
                    if st.button(f"Avvia Generazione su '{st.session_state.selected_sheet}'", key="generate_button"):
                        generate_judgments_callback(df_to_complete, giudizio_col, st.session_state.selected_sheet)

# SEZIONE 4: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================

st.markdown("---")
st.subheader("Stato dell'Operazione e Download")
status_container = st.container()
with status_container:
    for message in st.session_state.status_messages:
        if "Errore" in message:
            st.info(message) # st.info instead of st.error for a cleaner look in the app
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
        file_name=f"Giudizi_Generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

