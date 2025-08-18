# app.py - Orchestratore principale con gestione degli errori migliorata

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
import traceback # Importa il modulo traceback per ottenere una traccia dell'errore

# Importiamo il modulo che contiene la logica per leggere e preparare i file Excel.
# Questo modulo si occupa del troncamento e della preparazione dei dati.
from excel_reader import load_and_prepare_excel

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

# Impostiamo il titolo della pagina e l'icona per l'app Streamlit.
st.set_page_config(
    page_title="Generatore di Giudizi con IA",
    page_icon="🤖",
    layout="wide"
)

# Inizializziamo le variabili di stato della sessione per mantenere i dati tra le interazioni.
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame()
if 'file_to_process' not in st.session_state:
    st.session_state.file_to_process = None
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""

# ==============================================================================
# SEZIONE 3: FUNZIONI PER LA GESTIONE DEI FILE
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
        st.session_state.last_action_status = f"Errore nel salvataggio del file: {e}"
        return None

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE (UI) PRINCIPALE
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("---")

st.markdown("""
Questa applicazione ti aiuta a preparare un corpus di dati per il fine-tuning di un modello di linguaggio, partendo dai tuoi file Excel.
Il modello imparerà a generare giudizi basati sui dati che fornisci.
""")

# Crea due colonne per una migliore organizzazione dell'interfaccia.
col1, col2 = st.columns(2)

with col1:
    st.header("1. Carica i Tuoi File")
    st.markdown("Carica uno o più file Excel con i dati di valutazione (voti, note, ecc.) e la colonna 'Giudizio'.")
    uploaded_files = st.file_uploader(
        "Trascina e rilascia qui i tuoi file o clicca per caricarli.",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.file_to_process:
                st.session_state.file_to_process = file
                st.rerun()

with col2:
    st.header("2. Stato dell'Elaborazione")
    status_box = st.empty()
    if st.session_state.last_action_status:
        status_box.info(st.session_state.last_action_status)

# ==============================================================================
# SEZIONE 5: LOGICA DI ELABORAZIONE E GESTIONE DEL CORPUS
# ==============================================================================

if st.session_state.file_to_process:
    uploaded_file = st.session_state.file_to_process
    
    # Aggiorna lo stato dell'interfaccia.
    status_box.info(f"Elaborazione in corso: '{uploaded_file.name}'...")
    
    # Salva il file caricato e processalo.
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        try:
            # Chiama la funzione di elaborazione.
            # Aggiunto un feedback visivo più dettagliato che mostra il foglio in elaborazione.
            corpus_new_df = load_and_prepare_excel(file_path, status_box)
            
            # Se la funzione ritorna un DataFrame valido, uniscilo al corpus esistente.
            if not corpus_new_df.empty:
                st.session_state.corpus = pd.concat(
                    [st.session_state.corpus, corpus_new_df],
                    ignore_index=True
                )
                st.session_state.last_action_status = f"'{uploaded_file.name}' elaborato con successo! Righe aggiunte: {len(corpus_new_df)}"
            else:
                st.session_state.last_action_status = f"Impossibile elaborare il contenuto di '{uploaded_file.name}' o non contiene dati validi."
        except Exception as e:
            # Cattura qualsiasi errore durante il processo e mostra il traceback completo.
            st.session_state.last_action_status = f"Errore grave durante l'elaborazione di '{uploaded_file.name}':\n{e}\n\nTraceback:\n{traceback.format_exc()}"
            st.error(st.session_state.last_action_status)
    
    # Rimuove la variabile di stato per evitare una riesecuzione involontaria.
    del st.session_state.file_to_process
    st.rerun()

# ==============================================================================
# SEZIONE: VISUALIZZAZIONE E DOWNLOAD DEL CORPUS TOTALE
# ==============================================================================
st.write("---")
st.write("### Corpus Totale per il Fine-Tuning")
if not st.session_state.corpus.empty:
    st.dataframe(st.session_state.corpus, use_container_width=True)
    st.success(f"Il corpus totale contiene {len(st.session_state.corpus)} righe pronte per l'addestramento.")
    
    # Prepara il file in memoria per il download.
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

