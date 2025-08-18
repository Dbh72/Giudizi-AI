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
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# ==============================================================================
# SEZIONE 3: FUNZIONI PER I MESSAGGI DI STATO
# ==============================================================================

def add_status_message(message):
    """Aggiunge un messaggio alla lista di stato e lo stampa."""
    st.session_state.status_messages.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    st.info("\n".join(st.session_state.status_messages))
    print(message)

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("---")

# Tab per l'addestramento
tab1, tab2 = st.tabs(["Addestra Modello", "Genera Giudizi su File Excel"])

with tab1:
    st.header("1. Addestra il Modello")
    st.markdown("Carica uno o più file Excel per creare il corpus di addestramento. Il modello imparerà a generare giudizi basandosi sui tuoi dati.")
    
    uploaded_files = st.file_uploader(
        "Carica file Excel (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Prepara Dati per Addestramento"):
            try:
                # Svuota i dati precedenti e il corpus
                st.session_state.uploaded_files_data = {}
                st.session_state.corpus_df = pd.DataFrame()
                st.session_state.status_messages = []
                add_status_message("Avvio preparazione dei dati...")
                
                # Leggi e unisci i dati di tutti i file
                all_corpus_list = []
                for uploaded_file in uploaded_files:
                    add_status_message(f"Lettura del file '{uploaded_file.name}'...")
                    file_data = BytesIO(uploaded_file.getvalue())
                    # Corretto: Aggiunge 'uploaded_file.name' come secondo argomento
                    df = load_and_prepare_excel(file_data, uploaded_file.name)
                    
                    if not df.empty:
                        all_corpus_list.append(df)
                        add_status_message(f"Dati da '{uploaded_file.name}' caricati. Righe valide: {len(df)}")
                    else:
                        add_status_message(f"Attenzione: Nessun dato valido trovato nel file '{uploaded_file.name}'.")
                
                if all_corpus_list:
                    st.session_state.corpus_df = pd.concat(all_corpus_list, ignore_index=True)
                    add_status_message(f"Corpus totale creato con {len(st.session_state.corpus_df)} righe.")
                    st.success("Preparazione dati completata!")
                else:
                    st.warning("Nessun dato valido trovato in tutti i file caricati.")
                    
            except Exception as e:
                st.error(f"Errore durante la preparazione dei dati: {e}\n\nTraceback:\n{traceback.format_exc()}")
    
    st.markdown("---")
    if not st.session_state.corpus_df.empty:
        st.subheader("Avvia il Fine-Tuning")
        if st.button("Avvia Addestramento"):
            try:
                st.session_state.status_messages = []
                add_status_message("Avvio del fine-tuning. Il processo potrebbe richiedere del tempo.")
                
                # Esegue il fine-tuning utilizzando il modulo model_trainer.
                model_path = fine_tune_model(st.session_state.corpus_df)
                
                if model_path:
                    st.session_state.model_ready = True
                    st.success("Fine-tuning completato con successo!")
                    add_status_message(f"Modello salvato in: {model_path}")
                    st.balloons()
                    
                    # Offri il download del modello
                    st.download_button(
                        label="Scarica il modello addestrato (.zip)",
                        data=BytesIO(b'Dummy ZIP File Data'), # Sostituisci con la vera logica di compressione
                        file_name="modello_finetunato.zip",
                        mime="application/zip",
                    )
                else:
                    st.error("Addestramento fallito. Controlla i log per i dettagli.")
                    st.session_state.model_ready = False
            
            except Exception as e:
                st.error(f"Errore durante l'addestramento: {e}\n\nTraceback:\n{traceback.format_exc()}")

with tab2:
    st.header("2. Genera Giudizi su File")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello la compilerà e potrai scaricare il file aggiornato.")
    
    # Verifica che il modello sia stato addestrato
    if not st.session_state.model_ready:
        st.warning("Per generare giudizi, devi prima addestrare il modello nella sezione 'Addestra Modello'.")
    else:
        uploaded_file_to_complete = st.file_uploader(
            "Carica file Excel da completare",
            type=["xlsx", "xls", "xlsm"]
        )
        
        if uploaded_file_to_complete:
            try:
                df_temp = pd.read_excel(uploaded_file_to_complete, sheet_name=None)
                sheet_names = list(df_temp.keys())
                
                selected_sheet = st.selectbox(
                    "Seleziona il Foglio di Lavoro",
                    options=sheet_names,
                    key="sheet_selector"
                )
                
                st.session_state.selected_sheet = selected_sheet
                
                if st.button("Avvia Generazione su File"):
                    if not st.session_state.selected_sheet:
                        st.warning("Per favore, seleziona un foglio di lavoro.")
                        st.stop()
                        
                    # Ripulire i messaggi e il file completato
                    st.session_state.status_messages = []
                    st.session_state.process_completed_file = None
                    add_status_message(f"Avvio generazione per il foglio '{st.session_state.selected_sheet}'...")
                    
                    # Carica il modello fine-tuned
                    model, tokenizer = load_trained_model("modello_finetunato")
                    
                    if not model or not tokenizer:
                        st.error("Errore nel caricamento del modello. Assicurati che il percorso sia corretto.")
                        st.stop()
                    
                    # Legge il DataFrame del foglio selezionato
                    df_to_complete = pd.read_excel(uploaded_file_to_complete, sheet_name=st.session_state.selected_sheet)
                    giudizio_col = find_giudizio_column(df_to_complete)
                    
                    if not giudizio_col:
                        st.error("Non è stata trovata la colonna 'Giudizio' nel foglio selezionato.")
                        st.stop()
                        
                    # Esegue la generazione utilizzando il modulo judgment_generator
                    completed_df = generate_judgments_for_excel(
                        model, tokenizer, df_to_complete, giudizio_col, st.session_state.selected_sheet, "modello_finetunato"
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
        file_name=f"giudizi_aggiornati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
