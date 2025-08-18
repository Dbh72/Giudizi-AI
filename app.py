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
    st.session_state.file_to_process_corpus = []
if 'last_action_status_corpus' not in st.session_state:
    st.session_state.last_action_status_corpus = ""
if 'uploaded_file_generate' not in st.session_state:
    st.session_state.uploaded_file_generate = None
if 'excel_sheet_names' not in st.session_state:
    st.session_state.excel_sheet_names = []
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = ""

# ==============================================================================
# SEZIONE 3: FUNZIONI PER LA GESTIONE DEI FILE
# ==============================================================================

def save_uploaded_file(uploaded_file, directory="temp_uploads"):
    """Salva il file caricato dall'utente in una directory temporanea."""
    try:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        return None

def get_excel_sheet_names(file_path):
    """
    Ottiene i nomi dei fogli di lavoro di un file Excel.
    """
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        return workbook.sheetnames
    except Exception as e:
        return []

# La funzione di generazione del giudizio verrà implementata qui, dopo l'addestramento.
# Per ora, la lasciamo come commento per mantenere la struttura.
#
# def generate_giudizio_with_model(row_data):
#     """
#     Questa funzione conterrà la logica per chiamare il modello
#     addestrato (fine-tuned) e generare un giudizio.
#     """
#     pass

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE (UI) PRINCIPALE
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("---")

# Crea tre colonne per una migliore organizzazione dell'interfaccia.
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("1. Crea il tuo Corpus")
    st.markdown("Carica i tuoi file Excel per creare un corpus di dati per il fine-tuning del modello.")
    
    uploaded_files_corpus = st.file_uploader(
        "Trascina qui i file per l'addestramento.",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="file_uploader_corpus"
    )

    if uploaded_files_corpus:
        for file in uploaded_files_corpus:
            if file.name not in [f.name for f in st.session_state.file_to_process_corpus]:
                st.session_state.file_to_process_corpus.append(file)
                st.rerun()

    status_corpus_box = st.empty()
    if st.session_state.last_action_status_corpus:
        status_corpus_box.info(st.session_state.last_action_status_corpus)

    if st.session_state.file_to_process_corpus:
        uploaded_file = st.session_state.file_to_process_corpus.pop(0)
        status_corpus_box.info(f"Elaborazione in corso: '{uploaded_file.name}'...")
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            try:
                corpus_new_df = load_and_prepare_excel(file_path, status_corpus_box)
                if not corpus_new_df.empty:
                    st.session_state.corpus = pd.concat(
                        [st.session_state.corpus, corpus_new_df], ignore_index=True
                    )
                    st.session_state.last_action_status_corpus = f"'{uploaded_file.name}' elaborato con successo! Righe aggiunte: {len(corpus_new_df)}"
                else:
                    st.session_state.last_action_status_corpus = f"Impossibile elaborare '{uploaded_file.name}'."
            except Exception as e:
                st.session_state.last_action_status_corpus = f"Errore durante l'elaborazione di '{uploaded_file.name}':\n{e}\n\nTraceback:\n{traceback.format_exc()}"
                st.error(st.session_state.last_action_status_corpus)
        if st.session_state.file_to_process_corpus:
            st.rerun()

with col2:
    st.header("2. Visualizza Corpus")
    st.markdown("Qui puoi vedere il corpus di dati completo pronto per il fine-tuning.")
    
    if not st.session_state.corpus.empty:
        st.dataframe(st.session_state.corpus, use_container_width=True)
        st.success(f"Il corpus totale contiene {len(st.session_state.corpus)} righe pronte per l'addestramento.")
        
        corpus_buffer = BytesIO()
        with pd.ExcelWriter(corpus_buffer, engine='openpyxl') as writer:
            st.session_state.corpus.to_excel(writer, index=False, sheet_name='Corpus Totale')
        corpus_buffer.seek(0)
        
        st.download_button(
            label="Scarica il Corpus Totale",
            data=corpus_buffer,
            file_name="corpus_totale.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    else:
        st.info("Carica i file per iniziare a costruire il tuo corpus.")

with col3:
    st.header("3. Genera Giudizi")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota e il modello la compilerà.")

    # Widget per il caricamento del file da completare
    uploaded_file_generate = st.file_uploader(
        "Trascina qui il file da completare.",
        type=["xlsx", "xls", "xlsm"],
        key="file_uploader_generate"
    )

    # Gestione dello stato del file caricato
    if uploaded_file_generate:
        # Se viene caricato un nuovo file, resettiamo lo stato
        if st.session_state.uploaded_file_generate and uploaded_file_generate.name != st.session_state.uploaded_file_generate.name:
            st.session_state.excel_sheet_names = []
            st.session_state.process_completed_file = None
            st.session_state.generation_status = ""

        st.session_state.uploaded_file_generate = uploaded_file_generate
        # Salviamo il file temporaneamente e otteniamo i nomi dei fogli
        file_path_gen = save_uploaded_file(uploaded_file_generate, directory="temp_generate")
        if not st.session_state.excel_sheet_names:
            st.session_state.excel_sheet_names = get_excel_sheet_names(file_path_gen)
            if not st.session_state.excel_sheet_names:
                st.warning("Nessun foglio trovato. Assicurati che il file non sia protetto.")
    
    # Se i nomi dei fogli sono stati trovati, mostriamo il menu a tendina
    if st.session_state.excel_sheet_names:
        selected_sheet = st.selectbox(
            "Seleziona il Foglio di Lavoro",
            options=st.session_state.excel_sheet_names,
            key="sheet_selector"
        )
        
        # Bottone per avviare il processo di generazione
        if st.button("Avvia Generazione su File", use_container_width=True):
            status_box_gen = st.empty()
            status_box_gen.info("Inizio del processo di generazione...")
            try:
                # Carica il file e il foglio selezionato
                df_to_complete = pd.read_excel(st.session_state.uploaded_file_generate, sheet_name=selected_sheet)
                
                # Trova la colonna "Giudizio" (case-insensitive)
                giudizio_col = next((col for col in df_to_complete.columns if 'giudizio' in str(col).lower()), None)
                if not giudizio_col:
                    st.error("Colonna 'Giudizio' non trovata nel foglio selezionato.")
                    raise ValueError("Colonna 'Giudizio' mancante.")

                # Trova le altre colonne che fungeranno da input per il modello
                other_cols = [col for col in df_to_complete.columns if col != giudizio_col]
                
                # Itera sulle righe del DataFrame e genera i giudizi solo per le celle vuote
                for index, row in df_to_complete.iterrows():
                    # Controlla se il giudizio è vuoto (NaN o stringa vuota)
                    if pd.isna(row[giudizio_col]) or str(row[giudizio_col]).strip() == "":
                        # Prepara il prompt per il modello concatenando i dati delle altre colonne
                        prompt_parts = [f"{col}: {str(row[col]).strip()}" for col in other_cols if pd.notna(row[col]) and str(row[col]).strip()]
                        prompt_text = " ".join(prompt_parts)
                        
                        # Simula la chiamata al modello
                        # Questa parte dovrà essere sostituita con la logica di chiamata al modello addestrato.
                        # Per ora, usiamo una stringa di placeholder.
                        new_giudizio = "Giudizio simulato." 
                        
                        # Aggiorna il DataFrame con il giudizio generato
                        df_to_complete.at[index, giudizio_col] = new_giudizio
                        status_box_gen.info(f"Giudizio generato per la riga {index + 1}.")
                        
                # Salviamo il DataFrame completato nello stato della sessione
                st.session_state.process_completed_file = df_to_complete
                st.session_state.generation_status = "Generazione completata con successo!"
            except Exception as e:
                st.session_state.generation_status = f"Errore durante la generazione: {e}\n\nTraceback:\n{traceback.format_exc()}"
                st.error(st.session_state.generation_status)

    # Se il processo è completato, mostriamo un messaggio di successo e il bottone per il download
    if st.session_state.generation_status:
        st.success(st.session_state.generation_status)
        if st.session_state.process_completed_file is not None:
            st.write("### Scarica il file completato")
            output_buffer = BytesIO()
            # Salviamo il DataFrame aggiornato in un buffer di memoria
            with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=selected_sheet)
            output_buffer.seek(0)
            st.download_button(
                label="Scarica il file aggiornato",
                data=output_buffer,
                file_name=f"giudizi_aggiornati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
