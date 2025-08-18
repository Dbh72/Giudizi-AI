# app.py - Orchestratore principale con funzionalità unificate

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
import traceback
import openpyxl

# Importiamo i moduli con la logica per la preparazione dei dati
# Assicurati di avere un file 'excel_reader.py' nella stessa directory.
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
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
# Aggiungi questa linea per inizializzare file_to_process_corpus come lista vuota.
if 'file_to_process_corpus' not in st.session_state:
    st.session_state.file_to_process_corpus = []

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
        st.error(f"Errore nel salvare il file: {e}")
        return None

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE (UI) E LOGICA PRINCIPALE
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("---")

# Visualizzazione dello stato corrente dell'applicazione.
if st.session_state.last_action_status:
    st.info(st.session_state.last_action_status)

# ==============================================================================
# SOTTO-SEZIONE: CARICAMENTO E PREPARAZIONE DEL CORPUS
# ==============================================================================
st.write("### Carica i File per il Corpus di Fine-Tuning")
st.write("Carica uno o più file Excel per costruire il corpus. I file devono contenere la colonna 'Giudizio'.")

uploaded_files = st.file_uploader(
    "Trascina e rilascia i file qui",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

if uploaded_files:
    # Aggiungi solo i nuovi file allo stato della sessione
    new_files = [file for file in uploaded_files if file.name not in [f.name for f in st.session_state.uploaded_files]]
    st.session_state.uploaded_files.extend(new_files)

    # Prepara la lista dei file da processare
    files_to_process = [f for f in st.session_state.uploaded_files if f.name not in st.session_state.file_to_process_corpus]
    
    if files_to_process:
        with st.spinner("Elaborazione dei nuovi file..."):
            for file in files_to_process:
                # Salva il file per il modulo 'excel_reader'
                temp_file_path = save_uploaded_file(file)
                if temp_file_path:
                    # Utilizza il modulo load_and_prepare_excel per elaborare il file
                    df_new = load_and_prepare_excel(temp_file_path)
                    
                    if not df_new.empty:
                        # Concatena i nuovi dati al corpus esistente
                        st.session_state.corpus = pd.concat([st.session_state.corpus, df_new], ignore_index=True)
                        st.session_state.last_action_status = f"File '{file.name}' aggiunto al corpus con successo!"
                        st.session_state.file_to_process_corpus.append(file.name)
                    else:
                         st.session_state.last_action_status = f"Impossibile elaborare il contenuto di '{file.name}' o non contiene dati validi."
                    
                    # Rimuovi il file temporaneo
                    os.remove(temp_file_path)

# ==============================================================================
# SOTTO-SEZIONE: VISUALIZZAZIONE E DOWNLOAD DEL CORPUS TOTALE
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
        file_name="corpus_totale.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Carica i file per iniziare a costruire il tuo corpus.")
    
# ==============================================================================
# SOTTO-SEZIONE: GENERAZIONE DI GIUDIZI DA UN FILE
# ==============================================================================
st.write("---")
st.write("### Genera Giudizi su un File Esistente")
st.write("Carica un file Excel con la colonna 'Giudizio' da completare. **Questa funzionalità richiede un modello già addestrato.**")
# Questa sezione è ancora da implementare, come discusso in precedenza.
st.warning("Funzionalità di generazione non implementata in questo script. Usa lo script basato su Gradio o implementa qui la logica del fine-tuning e della generazione.")

