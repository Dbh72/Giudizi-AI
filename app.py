# ==============================================================================
# SEZIONE 0: CONTROLLO DELLE VERSIONI DELLE LIBRERIE
# ==============================================================================
# Questo blocco di codice verifica le versioni delle librerie chiave
# utilizzate nel progetto e le stampa nel terminale per scopi di debug.
# Verrà eseguito all'avvio dell'app.
import importlib
import sys

# Lista delle librerie da controllare.
# Nota: "transformers" è corretto, "gradio" e "sentence-transformers" sono stati rimossi.
libraries_to_check = ['pandas', 'openpyxl', 'datasets', 'transformers', 'peft', 'torch', 'streamlit']

print("--- Versioni delle Librerie Chiave ---")
for lib_name in libraries_to_check:
    try:
        # Gestione speciale per le librerie con nomi modulo diversi
        # (se necessario, come per 'sentence-transformers', ma in questo caso non serve)
        module_name = lib_name.replace('-', '_')
        lib = importlib.import_module(module_name)
        print(f"{lib_name}: {lib.__version__}")
    except ImportError:
        print(f"{lib_name}: Non trovata")
    except AttributeError:
        print(f"{lib_name}: Attributo __version__ non trovato")

print("---------------------------------")
print(f"Versione di Python: {sys.version}")
print("---------------------------------")


# ==============================================================================
# File: app.py
# Orchestratore principale con interfaccia utente Streamlit.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo le librerie essenziali per l'applicazione.
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import traceback
import openpyxl
import warnings
from datetime import datetime

# Importiamo i moduli con la logica per la preparazione dei dati, l'addestramento e la generazione.
from excel_reader import load_and_prepare_excel, find_giudizio_column
from model_trainer import fine_tune_model
from judgment_generator import generate_judgments_for_excel, load_trained_model

# Ignoriamo i FutureWarning per mantenere la console pulita
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

st.set_page_config(
    page_title="Generatore di Giudizi con IA",
    page_icon="🤖",
    layout="wide"
)

# Inizializziamo le variabili di stato della sessione per mantenere i dati.
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'fine_tuning_state' not in st.session_state:
    st.session_state.fine_tuning_state = {"status": "In attesa...", "last_checkpoint": None}
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'finetuned_model_path' not in st.session_state:
    st.session_state.finetuned_model_path = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# ==============================================================================
# SEZIONE 3: INTERFACCIA UTENTE PER IL FINE-TUNING
# ==============================================================================

st.title("🤖 Generatore di Giudizi con IA")
st.markdown("Questa applicazione ti aiuta a generare giudizi automatici utilizzando un modello di linguaggio fine-tuned su tuoi dati.")
st.markdown("---")

st.subheader("1. Addestramento del Modello (Fine-Tuning)")
st.info("Carica uno o più file Excel contenenti esempi di giudizi. Il sistema utilizzerà questi dati per addestrare il modello.")

uploaded_files_for_corpus = st.file_uploader(
    "Carica file per il corpus (Excel .xlsx, .xls, .xlsm)",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

if uploaded_files_for_corpus:
    # Aggiungiamo solo i file nuovi al dizionario di sessione
    new_files_uploaded = False
    for file in uploaded_files_for_corpus:
        if file.name not in st.session_state.uploaded_files_data:
            st.session_state.uploaded_files_data[file.name] = file.getvalue()
            new_files_uploaded = True

    # Se ci sono file nuovi o il corpus è vuoto, lo ricostruiamo
    if new_files_uploaded or st.session_state.corpus_df.empty:
        st.session_state.status_messages.append("Caricamento file per il corpus in corso...")
        combined_df = pd.DataFrame()
        for file_name, file_data in st.session_state.uploaded_files_data.items():
            st.session_state.status_messages.append(f"Lavorazione del file '{file_name}'...")
            df = load_and_prepare_excel(BytesIO(file_data), progress_container=st.session_state.status_messages)
            if not df.empty:
                combined_df = pd.concat([combined_df, df], ignore_index=True)

        if not combined_df.empty:
            st.session_state.corpus_df = combined_df
            st.session_state.status_messages.append(f"Corpus creato con successo! Trovate {len(st.session_state.corpus_df)} righe valide.")
            st.success(f"Corpus creato con successo! Trovate {len(st.session_state.corpus_df)} righe valide.")
        else:
            st.error("Nessun dato valido trovato nei file caricati per il fine-tuning.")
            st.session_state.corpus_df = pd.DataFrame()

    if not st.session_state.corpus_df.empty:
        st.dataframe(st.session_state.corpus_df.head(), use_container_width=True)
        st.write(f"Totale righe nel corpus: **{len(st.session_state.corpus_df)}**")
        st.info(f"Stato attuale: {st.session_state.fine_tuning_state['status']}")
        
        if st.button("Avvia Fine-Tuning"):
            if not st.session_state.corpus_df.empty:
                st.session_state.status_messages.append("Avvio del processo di fine-tuning...")
                with st.spinner("Addestramento del modello in corso..."):
                    try:
                        # Qui avviene l'addestramento
                        st.session_state.finetuned_model_path = fine_tune_model(st.session_state.corpus_df, st.session_state.fine_tuning_state)
                        st.session_state.fine_tuning_state["status"] = "Completato"
                        st.success("Addestramento completato con successo!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"Errore nel fine-tuning: {e}")
                        st.error(f"Traceback:\n{traceback.format_exc()}")
                        st.session_state.fine_tuning_state["status"] = "Fallito"
            else:
                st.warning("Per favore, carica i file Excel per creare un corpus prima di avviare l'addestramento.")

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE PER LA GENERAZIONE DEI GIUDIZI
# ==============================================================================

st.markdown("---")
st.subheader("2. Generazione dei Giudizi")
st.info("Carica il file Excel da completare. Il modello userà le colonne esistenti per generare i giudizi.")

uploaded_file_to_complete = st.file_uploader(
    "Carica file Excel da completare",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=False,
    key="file_to_complete_uploader"
)

if uploaded_file_to_complete:
    try:
        df_to_complete = pd.read_excel(uploaded_file_to_complete, sheet_name=None)
        sheet_names = list(df_to_complete.keys())
        st.session_state.selected_sheet = st.selectbox(
            "Seleziona un foglio di lavoro",
            options=sheet_names
        )

        giudizio_col = find_giudizio_column(pd.read_excel(uploaded_file_to_complete, sheet_name=st.session_state.selected_sheet))
        if giudizio_col:
            st.info(f"Trovata la colonna per i giudizi: **'{giudizio_col}'**")
            st.write("Anteprima del file da completare:")
            st.dataframe(pd.read_excel(uploaded_file_to_complete, sheet_name=st.session_state.selected_sheet).head())
        else:
            st.warning("La colonna 'Giudizio' non è stata trovata. Impossibile procedere.")
            st.stop()

    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        st.stop()

    if st.button("Avvia Generazione"):
        if st.session_state.finetuned_model_path is None:
            st.warning("Per favore, addestra il modello prima di avviare la generazione.")
        else:
            try:
                st.session_state.status_messages.append("Caricamento del modello fine-tuned...")
                if st.session_state.model is None or st.session_state.tokenizer is None:
                    st.session_state.model, st.session_state.tokenizer = load_trained_model(st.session_state.finetuned_model_path)
                
                if st.session_state.model is None or st.session_state.tokenizer is None:
                    st.error("Impossibile caricare il modello. Assicurati che l'addestramento sia andato a buon fine.")
                    st.stop()
                
                with st.spinner("Generazione dei giudizi in corso..."):
                    df_to_complete_sheet = pd.read_excel(uploaded_file_to_complete, sheet_name=st.session_state.selected_sheet)
                    completed_df = generate_judgments_for_excel(
                        st.session_state.model, st.session_state.tokenizer, df_to_complete_sheet,
                        giudizio_col, st.session_state.selected_sheet, "modello_finetunato"
                    )
                    
                    st.session_state.process_completed_file = completed_df
                    st.success("Generazione completata con successo!")
                    st.balloons()
            
            except Exception as e:
                st.error(f"Errore durante l'operazione: {e}\n\nTraceback:\n{traceback.format_exc()}")

# SEZIONE 5: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================

st.markdown("---")
st.subheader("Stato dell'Operazione e Download")
status_container = st.container()
with status_container:
    for message in st.session_state.status_messages:
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
        file_name=f"Giudizi_generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
