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
# Inizializza le variabili di stato della sessione per mantenere i dati tra i rerun.
if 'excel_files' not in st.session_state:
    st.session_state.excel_files = []
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame()
if 'excel_content_history' not in st.session_state:
    st.session_state.excel_content_history = {}
if 'file_to_process' not in st.session_state:
    st.session_state.file_to_process = None

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE
# ==============================================================================

# Titolo e descrizione dell'app.
st.title("Giudizi-AI: Preparazione del Corpus per il Fine-Tuning")
st.markdown("""
Questa applicazione ti aiuta a preparare i tuoi dati in formato Excel
per il fine-tuning di un modello di linguaggio che genererà giudizi
per le valutazioni.
""")

# Componente per il caricamento dei file.
# L'utente può caricare uno o più file Excel.
uploaded_files = st.file_uploader(
    "Carica uno o più file Excel (xlsx, xls, xlsm)",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

# Gestisce il caricamento dei file.
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Aggiunge il file alla lista se non è già presente.
        if uploaded_file.name not in [f.name for f in st.session_state.excel_files]:
            st.session_state.excel_files.append(uploaded_file)
            st.session_state.last_action_status = f"File '{uploaded_file.name}' caricato con successo!"
            # Imposta la variabile per l'elaborazione al prossimo rerun.
            st.session_state.file_to_process = uploaded_file.name
            st.rerun()

st.write("---")

# Mostra lo stato dell'ultima azione.
if st.session_state.last_action_status:
    st.info(st.session_state.last_action_status)

# Mostra la lista dei file caricati e offre la possibilità di rimuoverli.
st.write("### File Caricati:")
if st.session_state.excel_files:
    for idx, f in enumerate(st.session_state.excel_files):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.write(f"- {f.name}")
        with col2:
            # Pulsante per rimuovere un file dalla lista.
            if st.button("Rimuovi", key=f"remove_file_{idx}"):
                st.session_state.excel_files.pop(idx)
                # Rimuove anche i dati dal corpus e dalla history.
                if f.name in st.session_state.corpus:
                    st.session_state.corpus = st.session_state.corpus.drop(
                        st.session_state.corpus[st.session_state.corpus.iloc[:, 0] == f.name].index
                    )
                st.session_state.excel_content_history.pop(f.name, None)
                st.session_state.last_action_status = f"File '{f.name}' rimosso."
                st.rerun()
else:
    st.write("Nessun file caricato.")

st.write("---")

# ==============================================================================
# SEZIONE: LOGICA DI ELABORAZIONE (DOPO IL RERUN)
# ==============================================================================
if "file_to_process" in st.session_state and st.session_state.file_to_process:
    file_to_process_name = st.session_state["file_to_process"]
    # Trova il file caricato tra quelli nella sessione.
    uploaded_file = next((f for f in st.session_state.excel_files if f.name == file_to_process_name), None)

    if uploaded_file:
        with st.spinner(f"Elaborazione di {uploaded_file.name}..."):
            try:
                # Salva il file su disco per permettere al reader di accedervi.
                file_path = save_uploaded_file(uploaded_file)
                # Chiama la funzione per leggere l'Excel e preparare i dati.
                extracted_content = load_and_prepare_excel(file_path)
                
                if not extracted_content.empty:
                    # Se il corpus è vuoto, inizializzalo con il primo file.
                    if st.session_state.corpus.empty:
                        st.session_state.corpus = extracted_content
                    else:
                        # Altrimenti, aggiungi i nuovi dati.
                        st.session_state.corpus = pd.concat([st.session_state.corpus, extracted_content], ignore_index=True)
                    
                    st.session_state.excel_content_history[uploaded_file.name] = extracted_content
                    st.session_state.last_action_status = f"Contenuto di '{uploaded_file.name}' elaborato e aggiunto al corpus."
                else:
                    st.session_state.last_action_status = f"Impossibile elaborare il contenuto di '{uploaded_file.name}' o non contiene dati validi."
            except Exception as e:
                st.session_state.last_action_status = f"Errore nell'elaborazione di '{uploaded_file.name}': {e}"
    
    # Rimuove la variabile di stato per evitare una riesecuzione involontaria.
    del st.session_state.file_to_process
    st.rerun()

# ==============================================================================
# SEZIONE: VISUALIZZAZIONE E DOWNLOAD DEL CORPUS TOTALE
# ==============================================================================
st.write("---")
st.write("### Corpus Totale per il Fine-Tuning")
if not st.session_state.corpus.empty:
    st.dataframe(st.session_state.corpus)
    st.success(f"Il corpus totale contiene {len(st.session_state.corpus)} righe pronte per l'addestramento.")
    
    # Prepara il file in memoria per il download.
    corpus_buffer = BytesIO()
    with pd.ExcelWriter(corpus_buffer, engine='openpyxl') as writer:
        st.session_state.corpus.to_excel(writer, index=False, sheet_name='Corpus Totale')
    corpus_buffer.seek(0)
    
    st.download_button(
        label="Scarica il Corpus Totale",
        data=corpus_buffer,
        file_name="corpus_fine_tuning.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Carica i file Excel per iniziare a costruire il corpus.")
