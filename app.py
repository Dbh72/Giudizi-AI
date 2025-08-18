# app.py - Orchestratore principale e interfaccia Streamlit

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
# SEZIONE 2: FUNZIONI AUSILIARIE
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
# Inizializziamo le variabili di stato della sessione per mantenere i dati tra i rerun.
if 'excel_files' not in st.session_state:
    st.session_state.excel_files = []
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame()
if 'excel_content_history' not in st.session_state:
    st.session_state.excel_content_history = {}
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []


# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE STREAMLIT
# ==============================================================================
st.title("🤖 Giudizi-AI: Generazione e Analisi")

# Sidebar per il caricamento dei file
st.sidebar.header("Carica i tuoi file Excel")
uploaded_files = st.sidebar.file_uploader(
    "Trascina e rilascia i file qui:",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

# Gestisce il caricamento dei file e aggiorna la session_state
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.uploaded_file_names:
            st.session_state.excel_files.append(uploaded_file)
            st.session_state.uploaded_file_names.append(uploaded_file.name)
    
    st.sidebar.success("File caricati correttamente!")

# Elenco dei file caricati
st.sidebar.header("File Caricati")
if st.session_state.excel_files:
    for i, file in enumerate(st.session_state.excel_files):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.write(f"- {file.name}")
        with col2:
            if st.button("Elabora", key=f"elabora_{i}"):
                st.session_state.file_to_process = file.name
                st.rerun()

# Stato dell'ultima azione
if st.session_state.last_action_status:
    st.info(st.session_state.last_action_status)
    st.session_state.last_action_status = "" # Resetta il messaggio dopo la visualizzazione

st.write("---")

# ==============================================================================
# SEZIONE 5: LOGICA DI ELABORAZIONE (DOPO IL RERUN)
# ==============================================================================
# Questa sezione viene eseguita solo se un file è stato selezionato per l'elaborazione.
if "file_to_process" in st.session_state:
    file_to_process_name = st.session_state["file_to_process"]
    # Trova il file caricato nella lista della session_state
    uploaded_file = next((f for f in st.session_state.excel_files if f.name == file_to_process_name), None)

    if uploaded_file:
        with st.spinner(f"Elaborazione di {uploaded_file.name}..."):
            try:
                # Chiamiamo la funzione per leggere l'Excel e prepararne i dati
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    # Passiamo il file_path alla funzione di lettura
                    extracted_content = load_and_prepare_excel(file_path)

                    if not extracted_content.empty:
                        # Aggiorniamo la session_state con il nuovo contenuto
                        st.session_state.excel_content_history[uploaded_file.name] = extracted_content
                        
                        # Uniamo i dati di tutti i file per creare un corpus unificato
                        # Usiamo `pd.concat` per unire i DataFrame
                        if st.session_state.corpus.empty:
                            st.session_state.corpus = extracted_content
                        else:
                            st.session_state.corpus = pd.concat([st.session_state.corpus, extracted_content], ignore_index=True).drop_duplicates()

                        st.session_state.last_action_status = f"Contenuto di '{uploaded_file.name}' elaborato con successo! {len(extracted_content)} nuove righe aggiunte al corpus."
                    else:
                         st.session_state.last_action_status = f"Impossibile elaborare il contenuto di '{uploaded_file.name}' o non contiene dati validi."
                else:
                    st.session_state.last_action_status = f"Errore: File '{uploaded_file.name}' non salvato correttamente."
            except Exception as e:
                st.session_state.last_action_status = f"Errore nell'elaborazione di '{uploaded_file.name}': {e}"
    
    # Rimuove la variabile di stato per evitare una riesecuzione involontaria.
    del st.session_state.file_to_process
    st.rerun()

# ==============================================================================
# SEZIONE 6: VISUALIZZAZIONE E DOWNLOAD DEL CORPUS TOTALE
# ==============================================================================
# Mostriamo il DataFrame con l'intero corpus unificato e permettiamo di scaricarlo.
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
        file_name="corpus_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ==============================================================================
# SEZIONE 7: FUNZIONALITÀ DI FINE-TUNING E GENERAZIONE
# ==============================================================================
st.write("---")
st.write("### Addestramento e Generazione")
st.warning("Funzionalità in fase di sviluppo. Questa parte del codice è disabilitata.")

# La sezione di fine-tuning e generazione è stata commentata per ora.
# Svilupperemo questa parte in un secondo momento, una volta che la preparazione
# del corpus sarà stabile.

# def fine_tune_model_gradio(uploaded_file, progress=gr.Progress()):
#     # Logica di fine-tuning
#     pass
# 
# def generate_judgments_for_excel_gradio(file_to_complete, sheet_name):
#     # Logica di generazione dei giudizi
#     pass
#
# st.write("... Codice di fine-tuning e generazione commentato ...")

# ==============================================================================
# SEZIONE 8: VISUALIZZAZIONE STORICO FILES
# ==============================================================================
st.write("---")
st.write("### Riepilogo File Elaborati")
for file_name, df in st.session_state.excel_content_history.items():
    st.subheader(f"Contenuto di '{file_name}'")
    st.dataframe(df)

