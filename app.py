# app.py - Orchestratore principale

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo tutte le librerie essenziali per l'applicazione.
# streamlit per la creazione dell'interfaccia utente web.
# pandas per la manipolazione dei dati in formato DataFrame.
# os per la gestione del file system (creazione di directory, percorsi).
# io.BytesIO per gestire i file in memoria senza scriverli su disco.
import streamlit as st
import pandas as pd
import os
from io import BytesIO

# Importiamo il modulo che contiene la logica per leggere e preparare i file Excel.
# Questo modulo si occupa del troncamento e della preparazione dei dati.
from excel_reader import load_and_prepare_excel

# ==============================================================================
# SEZIONE 2: FUNZIONI PER LA GESTIONE DEI FILE
# ==============================================================================

def save_uploaded_file(uploaded_file):
    """Salva il file caricato dall'utente in una directory temporanea."""
    try:
        # Creiamo una cartella temporanea per i file caricati se non esiste già.
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Errore nel salvataggio del file: {e}")
        return None

# ==============================================================================
# SEZIONE 3: INIZIALIZZAZIONE DELLO STATO DELLA SESSIONE
# ==============================================================================
# Inizializziamo le variabili di stato della sessione per mantenere i dati
# e lo stato dell'applicazione attraverso i rerun di Streamlit.
if 'excel_files' not in st.session_state:
    st.session_state.excel_files = []
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
if 'excel_content_history' not in st.session_state:
    st.session_state.excel_content_history = {}
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame() # Inizializziamo il corpus come un DataFrame vuoto

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE
# ==============================================================================
st.set_page_config(page_title="Giudizi-AI", layout="wide")

st.title("🤖 Giudizi-AI: Preparazione del Corpus di Dati")
st.markdown("Carica i tuoi file Excel per creare un corpus di dati unificato per il fine-tuning.")
st.write("---")

# Area per il caricamento dei file
uploaded_files = st.file_uploader(
    "Carica i tuoi file Excel",
    type=['xlsx', 'xls', 'xlsm'],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Aggiungiamo i file caricati allo stato della sessione, evitando duplicati
        if uploaded_file not in st.session_state.excel_files:
            st.session_state.excel_files.append(uploaded_file)

# Mostriamo lo stato attuale e i file caricati
if st.session_state.last_action_status:
    st.info(st.session_state.last_action_status)

if st.session_state.excel_files:
    st.write("### File Excel Caricati")
    file_names = [f.name for f in st.session_state.excel_files]
    selected_file_name = st.selectbox("Seleziona un file da elaborare:", file_names)
    
    col1, col2 = st.columns(2)
    with col1:
        process_button = st.button(f"Elabora {selected_file_name}")
    with col2:
        if st.button("Elimina file selezionato"):
            file_to_remove = next((f for f in st.session_state.excel_files if f.name == selected_file_name), None)
            if file_to_remove:
                st.session_state.excel_files.remove(file_to_remove)
                st.session_state.last_action_status = f"File {selected_file_name} eliminato con successo."
                # Rimuoviamo il file dal corpus e dalla storia dei contenuti se esiste
                if selected_file_name in st.session_state.excel_content_history:
                    del st.session_state.excel_content_history[selected_file_name]
                if selected_file_name in st.session_state.corpus.index:
                    st.session_state.corpus.drop(selected_file_name, inplace=True)
            st.rerun()

    if process_button:
        st.session_state.file_to_process = selected_file_name
        st.rerun()

st.write("---")

# SEZIONE: LOGICA DI ELABORAZIONE (DOPO IL RERUN)
# Questa sezione viene eseguita solo se un file è stato selezionato per l'elaborazione.
if "file_to_process" in st.session_state:
    file_to_process_name = st.session_state["file_to_process"]
    uploaded_file = next((f for f in st.session_state.excel_files if f.name == file_to_process_name), None)

    if uploaded_file:
        with st.spinner(f"Elaborazione di {uploaded_file.name}..."):
            try:
                # Chiamiamo la funzione per leggere l'Excel e preparare i dati.
                # Questa funzione gestirà il troncamento delle righe/colonne e la pulizia dei dati.
                file_path = save_uploaded_file(uploaded_file)
                extracted_content = load_and_prepare_excel(file_path)

                if extracted_content is not None and not extracted_content.empty:
                    # Uniamo i dati del file elaborato al corpus generale.
                    # pd.concat si occupa di unire i DataFrame in un unico corpus.
                    st.session_state.corpus = pd.concat([st.session_state.corpus, extracted_content], ignore_index=True)
                    st.session_state.excel_content_history[uploaded_file.name] = extracted_content
                    st.session_state.last_action_status = f"Contenuto di {uploaded_file.name} elaborato e aggiunto al corpus con successo!"
                    
                else:
                     st.session_state.last_action_status = f"Impossibile elaborare il contenuto di {uploaded_file.name} o non contiene dati validi."
            except Exception as e:
                st.session_state.last_action_status = f"Errore nell'elaborazione di {uploaded_file.name}: {e}"
        
        # Rimuove la variabile di stato per evitare una riesecuzione involontaria
        del st.session_state.file_to_process
        st.rerun()

# SEZIONE: VISUALIZZAZIONE E DOWNLOAD DEL CORPUS TOTALE
# Mostriamo il DataFrame con l'intero corpus unificato e permettiamo di scaricarlo.
st.write("---")
st.write("### Corpus Totale per il Fine-Tuning")
if not st.session_state.corpus.empty:
    st.dataframe(st.session_state.corpus)
    st.success(f"Il corpus totale contiene {len(st.session_state.corpus)} righe pronte per l'addestramento.")
    
    # Prepara il file in memoria per il download
    corpus_buffer = BytesIO()
    with pd.ExcelWriter(corpus_buffer, engine='openpyxl') as writer:
        st.session_state.corpus.to_excel(writer, index=False, sheet_name='Corpus Totale')
    corpus_buffer.seek(0)
    
    st.download_button(
        label="Scarica il Corpus Totale",
        data=corpus_buffer,
        file_name="corpus_totale.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Carica ed elabora i file per creare il corpus.")
