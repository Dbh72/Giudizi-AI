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
# Ho modificato l'import per usare il file excel_reader.py
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

# Funzione per aggiungere messaggi di stato alla sessione Streamlit
def progress_container(message, type="info"):
    st.session_state.status_messages.append({"message": message, "type": type})
    time.sleep(0.5)
    st.session_state.status_messages = st.session_state.status_messages
    st.rerun()

# Funzione per verificare se un modello esiste già.
def check_model_exists():
    return os.path.exists(os.path.join(OUTPUT_DIR, "final_model"))

# ==============================================================================
# SEZIONE 2: LAYOUT E INTERFACCIA UTENTE
# ==============================================================================

st.set_page_config(layout="wide")
st.title("Generatore di Giudizi per Alunni")

# Inizializza gli stati della sessione se non esistono
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []
if "process_completed_file" not in st.session_state:
    st.session_state.process_completed_file = None
if "model_available" not in st.session_state:
    st.session_state.model_available = check_model_exists()
if "fine_tuning_state" not in st.session_state:
    st.session_state.fine_tuning_state = "initial"
if "selected_sheet" not in st.session_state:
    st.session_state.selected_sheet = None


# Layout con due colonne
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Addestramento del Modello")
    st.markdown("---")

    st.markdown("Carica un file Excel con una colonna 'Giudizio' già compilata. I dati verranno usati per addestrare il modello.")

    train_file_input = st.file_uploader(
        "Carica file Excel per l'addestramento",
        type=['xlsx', 'xls', 'xlsm']
    )

    if st.button("Addestra il modello"):
        if train_file_input:
            progress_container("Avvio processo di addestramento...", "info")
            try:
                # Carica il dataframe dal file Excel
                train_df = er.read_and_prepare_data_from_excel(train_file_input, progress_container)
                
                if not train_df.empty:
                    # Costruisce/aggiorna il corpus di addestramento
                    corpus_df = cb.build_or_update_corpus(train_df, progress_container)
                    
                    if not corpus_df.empty:
                        # Addestra il modello
                        progress_container("Corpus creato con successo. Avvio fine-tuning...", "success")
                        mt.fine_tune_model(corpus_df, progress_container)
                        st.session_state.model_available = True
                    else:
                        progress_container("Impossibile creare il corpus. Controlla il file di addestramento.", "error")
                else:
                    progress_container("Il file di addestramento non contiene dati validi.", "error")

            except Exception as e:
                progress_container(f"Errore durante l'addestramento del modello: {e}", "error")
                progress_container(f"Traceback: {traceback.format_exc()}", "error")
        else:
            progress_container("Carica un file per l'addestramento prima di procedere.", "warning")
            

with col2:
    st.header("2. Generazione dei Giudizi")
    st.markdown("---")
    
    if st.session_state.model_available:
        st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna e potrai scaricare il file aggiornato.")
        
        generate_file_input = st.file_uploader(
            "Carica file Excel da completare",
            type=['xlsx', 'xls', 'xlsm'],
        )
        
        if generate_file_input:
            # Mostra i fogli di lavoro disponibili
            sheet_names = er.get_excel_sheet_names(generate_file_input)
            st.session_state.selected_sheet = st.selectbox(
                "Seleziona un foglio di lavoro",
                options=sheet_names
            )
            
            if st.button("Genera giudizi su file"):
                progress_container("Caricamento e avvio generazione giudizi...", "info")
                try:
                    # Carica il modello fine-tuned
                    model, tokenizer = jg.load_model(os.path.join(OUTPUT_DIR, "final_model"), progress_container)
                    
                    if model and tokenizer:
                        # Genera i giudizi
                        updated_df, _ = jg.generate_judgments_on_excel(
                            model, tokenizer, generate_file_input, st.session_state.selected_sheet, progress_container
                        )
                        st.session_state.process_completed_file = updated_df
                        progress_container("Processo di generazione completato. Puoi scaricare il file.", "success")
                    else:
                        progress_container("Impossibile caricare il modello. Addestralo prima di procedere.", "error")
                
                except Exception as e:
                    progress_container(f"Errore durante la generazione dei giudizi: {e}", "error")
                    progress_container(f"Traceback: {traceback.format_exc()}", "error")
    
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

