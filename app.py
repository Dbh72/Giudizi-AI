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
    Mostra un messaggio di progresso o stato all'interno di un contenitore
    Streamlit, con stili diversi in base al tipo di messaggio.
    
    Args:
        placeholder: Il contenitore (placeholder) Streamlit in cui mostrare il messaggio.
        message (str): Il messaggio da visualizzare.
        type (str): Il tipo di messaggio ('info', 'success', 'error', 'warning').
    """
    # Aggiunge il messaggio al log
    with open(os.path.join(OUTPUT_DIR, "Logs.txt"), "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {type.upper()}: {message}\n")
    
    # Mostra il messaggio nell'interfaccia utente
    if type == "info":
        placeholder.info(message, icon="ℹ️")
    elif type == "success":
        placeholder.success(message, icon="✅")
    elif type == "error":
        placeholder.error(message, icon="❌")
    elif type == "warning":
        placeholder.warning(message, icon="⚠️")

def create_model_zip(source_dir, output_filename):
    """
    Comprime una directory in un file zip.
    """
    shutil.make_archive(output_filename.replace(".zip", ""), 'zip', source_dir)

# ==============================================================================
# SEZIONE 1: IMPOSTAZIONE E INTERFACCIA UTENTE PRINCIPALE
# ==============================================================================

# Configurazione della pagina
st.set_page_config(
    page_title="Generatore di Giudizi",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👨‍🏫 Generatore Automatico di Giudizi")
st.markdown("---")

# Inizializza gli stati di sessione se non esistono
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'corpus_created' not in st.session_state:
    st.session_state.corpus_created = False
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = ""

# Assicurati che la directory di output esista
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Carica corpus esistente all'avvio
if st.session_state.corpus_df.empty:
    st.session_state.corpus_df = cb.load_corpus(os.path.join(OUTPUT_DIR, CORPUS_FILE))

with st.expander("Gestione del Corpus di Addestramento"):
    st.write("Puoi caricare uno o più file per costruire il corpus. Il corpus è il set di dati con cui il modello imparerà a generare i giudizi.")
    
    uploaded_corpus_file = st.file_uploader(
        "Carica il file Excel per il corpus (.xlsx, .xls, .xlsm)", 
        type=["xlsx", "xls", "xlsm"],
        key="corpus_uploader"
    )

    if uploaded_corpus_file:
        try:
            progress_placeholder = st.empty()
            progress_container(progress_placeholder, "Caricamento del file del corpus...", "info")
            corpus_data = er.load_excel_with_sheets(uploaded_corpus_file)
            progress_container(progress_placeholder, "File caricato con successo. Processo in corso...", "info")
            
            # Utilizza il primo foglio come default
            sheet_name = list(corpus_data.keys())[0]
            df_new_data = corpus_data[sheet_name]
            
            # Converti il DataFrame in un formato adatto per il corpus
            df_corpus = er.convert_to_corpus_format(df_new_data)
            
            # Aggiorna il corpus esistente o ne crea uno nuovo
            cb.build_corpus(df_corpus, os.path.join(OUTPUT_DIR, CORPUS_FILE))
            st.session_state.corpus_df = cb.load_corpus(os.path.join(OUTPUT_DIR, CORPUS_FILE))
            st.session_state.corpus_created = True
            progress_container(progress_placeholder, f"Corpus di addestramento aggiornato con successo. Totale righe: {len(st.session_state.corpus_df)}", "success")

        except Exception as e:
            progress_container(progress_placeholder, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
            st.error("Errore nel caricamento del file. Controlla il formato e riprova.")

    st.write("### Stato del Corpus")
    if st.session_state.corpus_df.empty:
        st.warning("Il corpus non è ancora stato creato. Carica un file per iniziare.")
    else:
        st.success(f"Corpus pronto. Numero di esempi di addestramento: {len(st.session_state.corpus_df)}")
        st.write("Anteprima del corpus:")
        st.dataframe(st.session_state.corpus_df.head(5))

st.markdown("---")

# ==============================================================================
# SEZIONE 2: ADDESTRAMENTO E GENERAZIONE DEI GIUDIZI
# ==============================================================================

st.header("1. Addestramento del Modello")
st.info("Questa operazione richiede molto tempo e risorse. Assicurati di avere il corpus pronto.")

if st.session_state.corpus_created:
    if st.button("Avvia Addestramento (Fine-Tuning)"):
        with st.spinner("Addestramento in corso... potrebbe volerci del tempo."):
            try:
                # Inizializza i placeholder per i log
                progress_placeholder_train = st.empty()
                progress_container(progress_placeholder_train, "Preparazione per l'addestramento...", "info")
                
                # Resetta il modello e il tokenizer per evitare problemi di cache
                st.session_state.trained_model = None
                st.session_state.tokenizer = None
                
                # Avvia l'addestramento
                st.session_state.trained_model, st.session_state.tokenizer = mt.train_model(
                    st.session_state.corpus_df, progress_container
                )
                
                # Salva il modello e il tokenizer
                final_model_path = os.path.join(OUTPUT_DIR, "final_model")
                st.session_state.trained_model.save_pretrained(final_model_path)
                st.session_state.tokenizer.save_pretrained(final_model_path)
                
                progress_container(progress_placeholder_train, "Addestramento completato con successo!", "success")
                st.balloons()
            except Exception as e:
                progress_container(progress_placeholder_train, f"Errore durante l'addestramento: {e}", "error")
                st.error("Si è verificato un errore durante l'addestramento. Controlla i log per i dettagli.")
                
else:
    st.warning("Devi creare un corpus di addestramento nella sezione 'Gestione del Corpus' prima di poter addestrare un modello.")

st.markdown("---")
st.header("2. Generazione dei Giudizi")

if st.session_state.trained_model:
    uploaded_process_file = st.file_uploader(
        "Carica il file Excel da cui generare i giudizi", 
        type=["xlsx", "xls", "xlsm"], 
        key="process_uploader"
    )

    if uploaded_process_file:
        try:
            progress_placeholder_generate = st.empty()
            progress_container(progress_placeholder_generate, "Caricamento del file da processare...", "info")
            df_to_process, selected_sheet = er.read_excel_to_df(uploaded_process_file, progress_container)
            
            st.session_state.process_completed_file = jg.generate_judgments(
                df_to_process,
                st.session_state.trained_model,
                st.session_state.tokenizer,
                progress_container
            )
            st.session_state.selected_sheet = selected_sheet
            
            progress_container(progress_placeholder_generate, "Generazione dei giudizi completata con successo!", "success")
            st.success("Operazione completata! Puoi scaricare il file aggiornato qui sotto.")

        except Exception as e:
            progress_container(progress_placeholder_generate, f"Errore nel caricamento del file. Controlla il formato e riprova. {e}", "error")
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
            label="Scarica Log",
            data=fp,
            file_name="Logs.txt",
            mime="text/plain"
        )
