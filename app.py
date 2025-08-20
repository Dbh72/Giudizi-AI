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

def progress_container(status_placeholder, message, type):
    """
    Gestisce l'aggiornamento del placeholder di stato con un messaggio
    colorato a seconda del tipo (info, success, warning, error).
    """
    if type == "info":
        status_placeholder.info(message)
    elif type == "success":
        status_placeholder.success(message)
    elif type == "warning":
        status_placeholder.warning(message)
    elif type == "error":
        status_placeholder.error(message)

def train_model(progress_container, corpus_df):
    """
    Avvia il processo di addestramento del modello e aggiorna lo stato.
    """
    progress_container("Addestramento del modello in corso...", "info")
    st.session_state.fine_tuned_model, st.session_state.tokenizer = mt.train_and_save_model(
        corpus_df=corpus_df,
        progress_container=progress_container
    )
    if st.session_state.fine_tuned_model is not None:
        progress_container("Modello addestrato e salvato con successo!", "success")
        st.session_state.model_trained = True

def generate_judgments_and_save(file_object, selected_sheet, progress_container):
    """
    Genera i giudizi per il file caricato e salva il risultato.
    """
    try:
        if file_object is None:
            progress_container("Nessun file caricato per la generazione.", "error")
            return

        # Assicuriamoci che il puntatore del file sia all'inizio prima di leggere
        file_object.seek(0)

        progress_container("Lettura del file per la generazione...", "info")
        df_to_process = er.read_and_prepare_data_from_excel(
            file_object=file_object,
            sheet_names=[selected_sheet],
            progress_container=progress_container,
            training_mode=False
        )

        if df_to_process.empty:
            progress_container("Nessuna riga da completare trovata nel file.", "warning")
            st.session_state.process_completed_file = None
            return

        progress_container(f"Trovate {len(df_to_process)} righe da processare.", "info")
        
        # Genera i giudizi utilizzando il modello
        st.session_state.process_completed_file = jg.generate_judgments(
            df=df_to_process,
            model=st.session_state.fine_tuned_model,
            tokenizer=st.session_state.tokenizer,
            sheet_name=selected_sheet,
            progress_container=progress_container
        )
        
        progress_container("Generazione dei giudizi completata. Il file è pronto per il download.", "success")
        st.session_state.process_completed = True

    except Exception as e:
        progress_container(f"Errore critico durante la generazione dei giudizi: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        st.session_state.process_completed_file = None
        st.session_state.process_completed = False

# ==============================================================================
# SEZIONE 2: LAYOUT E COMPONENTI DI STREAMLIT
# ==============================================================================

# Configurazione della pagina
st.set_page_config(
    page_title="Generatore di Giudizi AI",
    page_icon="🤖",
    layout="wide"
)

# Inizializzazione degli stati di sessione
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "fine_tuned_model" not in st.session_state:
    st.session_state.fine_tuned_model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "corpus_df" not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "process_completed_file" not in st.session_state:
    st.session_state.process_completed_file = None
if "process_completed" not in st.session_state:
    st.session_state.process_completed = False
if "selected_sheet" not in st.session_state:
    st.session_state.selected_sheet = None

st.title("Generatore di Giudizi per la Scuola 📚🤖")
st.markdown("---")

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
st.header("1. Addestramento del Modello")
st.markdown("Carica un file Excel con giudizi esistenti per addestrare il modello. Il file può avere più fogli, ma verranno utilizzati solo quelli che contengono la parola 'Giudizio' nell'intestazione.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_training_file = st.file_uploader(
        "Carica il file Excel di addestramento (es. .xlsx, .xlsm)",
        type=["xlsx", "xlsm", "xls"]
    )

with col2:
    if st.button("Carica e Aggiorna Corpus"):
        if uploaded_training_file is not None:
            status_placeholder_load = st.empty()
            try:
                # Assicuriamoci che il puntatore del file sia all'inizio prima di leggere
                uploaded_training_file.seek(0)
                
                sheet_names = er.get_excel_sheet_names(uploaded_training_file)
                if not sheet_names:
                    progress_container(status_placeholder_load, "Errore: Nessun foglio di lavoro valido trovato nel file.", "error")
                else:
                    progress_container(status_placeholder_load, f"Fogli trovati: {', '.join(sheet_names)}", "info")
                    
                    # Prepara i dati da tutti i fogli
                    st.session_state.corpus_df = cb.build_or_update_corpus(
                        er.read_and_prepare_data_from_excel(
                            uploaded_training_file, 
                            sheet_names=sheet_names, 
                            progress_container=status_placeholder_load
                        ),
                        progress_container=status_placeholder_load
                    )
                    st.session_state.uploaded_file_name = uploaded_training_file.name
                    if not st.session_state.corpus_df.empty:
                        progress_container(status_placeholder_load, f"Corpus aggiornato con {len(st.session_state.corpus_df)} righe totali.", "success")
            except Exception as e:
                progress_container(status_placeholder_load, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
                st.error("Errore nel caricamento del file. Controlla il formato e riprova.")
        else:
            st.warning("Per favore, carica prima un file di addestramento.")

col_train, col_load_model = st.columns([1,1])

with col_train:
    if st.button("Avvia Addestramento"):
        if not st.session_state.corpus_df.empty:
            status_placeholder_train = st.empty()
            train_model(status_placeholder_train, st.session_state.corpus_df)
        else:
            st.warning("Carica o addestra il modello prima di procedere.")

with col_load_model:
    if st.button("Carica Modello Esistente"):
        if os.path.exists(os.path.join(OUTPUT_DIR, "final_model")):
            status_placeholder_load_model = st.empty()
            st.session_state.fine_tuned_model, st.session_state.tokenizer = mt.load_fine_tuned_model(status_placeholder_load_model)
            if st.session_state.fine_tuned_model is not None:
                st.session_state.model_trained = True
                st.session_state.uploaded_file_name = "Modello caricato da file"
        else:
            st.warning("Nessun modello salvato trovato. Addestra prima un nuovo modello.")

# Visualizzazione stato corrente
if st.session_state.model_trained:
    st.success("Stato: Modello caricato e pronto per la generazione di giudizi!")
else:
    st.warning("Stato: Nessun modello caricato. Addestra o carica un modello per procedere.")

if not st.session_state.corpus_df.empty:
    st.info(f"Corpus di addestramento corrente: {len(st.session_state.corpus_df)} righe. Aggiornato da: {st.session_state.uploaded_file_name}")

st.markdown("---")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
st.header("2. Generazione dei Giudizi")

if st.session_state.model_trained:
    uploaded_process_file = st.file_uploader(
        "Carica il file Excel con le righe da completare (es. .xlsx, .xlsm)",
        type=["xlsx", "xlsm", "xls"],
        key="process_uploader"
    )

    if uploaded_process_file:
        status_placeholder_generate = st.empty()
        
        # Assicuriamoci che il puntatore del file sia all'inizio prima di leggere
        uploaded_process_file.seek(0)
        
        sheet_names = er.get_excel_sheet_names(uploaded_process_file)
        if not sheet_names:
            progress_container(status_placeholder_generate, "Errore: Nessun foglio di lavoro valido trovato nel file.", "error")
        else:
            selected_sheet = st.selectbox(
                "Seleziona il foglio di lavoro da processare:",
                sheet_names,
                key="sheet_selector"
            )
            st.session_state.selected_sheet = selected_sheet
            if st.button("Genera Giudizi"):
                generate_judgments_and_save(uploaded_process_file, selected_sheet, status_placeholder_generate)
else:
    st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

st.markdown("---")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.header("3. Stato e Download")

if st.session_state.process_completed:
    st.success("Generazione completata! Il file è pronto per il download.")
    if st.session_state.process_completed_file is not None:
        st.write("### Scarica il file completato")
        
        # Creiamo un buffer in memoria per il file Excel
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            # Per salvare, dobbiamo usare un solo foglio, come concordato
            st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
        output_buffer.seek(0)
        
        st.download_button(
            label="Scarica il file aggiornato",
            data=output_buffer,
            file_name=f"Giudizi_Generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
