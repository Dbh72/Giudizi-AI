# app.py - Orchestratore principale con funzionalità unificate

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo tutte le librerie essenziali per l'applicazione.
# streamlit per la creazione dell'interfaccia utente web.
# pandas per la manipolazione dei dati in formato DataFrame.
# os per la gestione del file system (creazione di directory, percorsi).
# io.BytesIO per gestire i file in memoria senza scriverli su disco.
# openpyxl per la lettura e scrittura di file Excel.
# traceback per la gestione degli errori.
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
if 'file_to_process_corpus' not in st.session_state:
    st.session_state.file_to_process_corpus = pd.DataFrame()
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
if 'fine_tuning_state' not in st.session_state:
    st.session_state.fine_tuning_state = "ready"
if 'download_file_path' not in st.session_state:
    st.session_state.download_file_path = None
if 'fine_tune_file_input' not in st.session_state:
    st.session_state.fine_tune_file_input = None

# ==============================================================================
# SEZIONE 3: GESTIONE DEL CARICAMENTO DEI FILE
# ==============================================================================

def save_uploaded_file(uploaded_file):
    """Salva il file caricato dall'utente in una directory temporanea."""
    try:
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Errore nel salvataggio del file: {e}")
        return None

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("---")

# Crea una barra laterale per le opzioni
with st.sidebar:
    st.header("Opzioni")
    selected_option = st.radio("Seleziona una funzione:", ["Crea Corpus", "Genera Giudizi"])

# ==============================================================================
# SEZIONE 5: LOGICA DELLE FUNZIONALITÀ
# ==============================================================================

# Logica per la creazione del corpus
if selected_option == "Crea Corpus":
    st.subheader("Costruisci il tuo Corpus per il Fine-Tuning")
    st.markdown("Carica uno o più file Excel. Il sistema identificherà le colonne e creerà un dataset unico per l'addestramento.")

    uploaded_files_corpus = st.file_uploader(
        "Carica uno o più file Excel",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True
    )

    if uploaded_files_corpus:
        with st.spinner("Preparazione del corpus in corso..."):
            for uploaded_file in uploaded_files_corpus:
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    try:
                        new_data = load_and_prepare_excel(file_path)
                        if not new_data.empty:
                            st.session_state.corpus = pd.concat([st.session_state.corpus, new_data], ignore_index=True).drop_duplicates()
                            st.session_state.last_action_status = f"File '{uploaded_file.name}' elaborato e aggiunto al corpus."
                        else:
                            st.session_state.last_action_status = f"Impossibile elaborare il contenuto di '{uploaded_file.name}' o non contiene dati validi."
                    except Exception as e:
                        st.session_state.last_action_status = f"Errore nell'elaborazione di '{uploaded_file.name}': {e}"
                        st.error(st.session_state.last_action_status)

    if not st.session_state.corpus.empty:
        st.write("---")
        st.write("### Corpus Totale per il Fine-Tuning")
        st.dataframe(st.session_state.corpus)
        st.success(f"Il corpus totale contiene {len(st.session_state.corpus)} righe pronte per l'addestramento.")

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

    if st.session_state.last_action_status:
        st.markdown(f"**Stato:** {st.session_state.last_action_status}")
        st.session_state.last_action_status = ""

# Logica per la generazione di giudizi
if selected_option == "Genera Giudizi":
    st.subheader("Genera Giudizi su un File Esistente")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' da completare. **Questa funzionalità richiede un modello già addestrato.**")
    
    # La logica di caricamento del modello e di fine-tuning non è inclusa in questo snippet.
    # Dovrai aggiungere qui il codice per caricare il modello addestrato.

    uploaded_file_generate = st.file_uploader(
        "Carica file Excel da completare",
        type=["xlsx", "xls", "xlsm"]
    )

    if uploaded_file_generate:
        file_path_generate = save_uploaded_file(uploaded_file_generate)
        if file_path_generate:
            st.session_state.process_file_path = file_path_generate
            try:
                # Leggiamo il file senza preparare il corpus, per ottenere i nomi dei fogli.
                workbook = openpyxl.load_workbook(file_path_generate)
                sheet_names = workbook.sheetnames
                st.session_state.sheet_names = sheet_names
            except Exception as e:
                st.error(f"Errore nella lettura del file per ottenere i fogli: {e}")
                st.session_state.sheet_names = []
    
    if 'sheet_names' in st.session_state and st.session_state.sheet_names:
        selected_sheet = st.selectbox(
            "Seleziona il Foglio di Lavoro da completare",
            options=st.session_state.sheet_names
        )

        if st.button("Avvia Generazione"):
            if selected_sheet and st.session_state.process_file_path:
                with st.spinner("Generazione dei giudizi in corso..."):
                    try:
                        # Leggiamo il file e il foglio selezionato
                        df_to_complete = pd.read_excel(st.session_state.process_file_path, sheet_name=selected_sheet)
                        
                        # Aggiungere qui la logica di caricamento del modello e generazione
                        # Questa è solo una simulazione del processo di generazione.
                        giudizio_col = "Giudizio" # Questo andrebbe cercato in modo dinamico
                        if giudizio_col not in df_to_complete.columns:
                            st.warning("Colonna 'Giudizio' non trovata. Impossibile procedere.")
                        else:
                            for index, row in df_to_complete.iterrows():
                                if pd.isna(row[giudizio_col]):
                                    # Genera il giudizio usando il modello
                                    # Esempio: generated_judgement = modello.generate(...)
                                    df_to_complete.loc[index, giudizio_col] = f"Giudizio generato per la riga {index + 1}."
                                st.session_state.generation_status = "Generazione completata con successo!"
                                st.session_state.process_completed_file = df_to_complete
                                
                    except Exception as e:
                        st.error(f"Errore durante la generazione: {e}\n\nTraceback:\n{traceback.format_exc()}")
            else:
                st.warning("Per favore, seleziona un foglio di lavoro.")
    
    if st.session_state.generation_status:
        st.success(st.session_state.generation_status)
        if st.session_state.process_completed_file is not None:
            st.write("### Scarica il file completato")
            output_buffer = BytesIO()
            with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=selected_sheet)
            output_buffer.seek(0)
            
            st.download_button(
                label="Scarica il file aggiornato",
                data=output_buffer,
                file_name=f"giudizi_aggiornati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

