# ==============================================================================
# File: app.py
# Orchestratore principale con funzionalità unificate
# ==============================================================================
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import traceback
import openpyxl

# Importiamo il modulo con la logica per la preparazione dei dati
# Assicurati che 'excel_reader.py' si trovi nella stessa directory.
from excel_reader import load_and_prepare_excel, find_giudizio_column

# ==============================================================================
# SEZIONE 1: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

# Impostiamo il titolo della pagina e l'icona per l'app Streamlit.
st.set_page_config(
    page_title="Generatore di Giudizi con IA",
    page_icon="🤖",
    layout="wide"
)

# Inizializziamo le variabili di stato della sessione per mantenere i dati tra le interazioni.
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}  # Dizionario per i dati dei file caricati
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'fine_tuning_state' not in st.session_state:
    st.session_state.fine_tuning_state = {"status": "In attesa", "progress": 0.0, "current_step": "In attesa..."}
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'excel_sheets' not in st.session_state:
    st.session_state.excel_sheets = []
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None

# ==============================================================================
# SEZIONE 2: LAYOUT E INTERFACCIA UTENTE
# ==============================================================================

st.title("🤖 Generatore di Giudizi con IA")
st.markdown("Questa applicazione permette di preparare un corpus di dati per il fine-tuning di un modello di linguaggio e di utilizzarlo per generare giudizi su nuovi file Excel.")

# =========================
# Scheda 'Prepara Corpus'
# =========================
with st.expander("📂 Prepara Corpus per Fine-Tuning", expanded=True):
    st.header("Carica File Excel per Addestramento")
    st.markdown("Carica i tuoi file Excel (`.xlsx`, `.xls`, `.xlsm`). Assicurati che contengano una colonna chiamata 'Giudizio'.")
    
    uploaded_files = st.file_uploader(
        "Seleziona uno o più file...",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True
    )
    
    # Se un file viene caricato, lo processiamo
    if uploaded_files:
        try:
            for file in uploaded_files:
                # Usiamo il nome del file come chiave per evitare duplicati
                st.session_state.uploaded_files_data[file.name] = file.read()
            
            # Ricostruiamo il corpus completo da tutti i file caricati
            full_corpus_list = []
            for file_name, file_data in st.session_state.uploaded_files_data.items():
                df = load_and_prepare_excel(BytesIO(file_data))
                full_corpus_list.append(df)
            
            # Uniamo tutti i DataFrame in un unico corpus
            if full_corpus_list and not all(df.empty for df in full_corpus_list):
                st.session_state.corpus_df = pd.concat(full_corpus_list, ignore_index=True)
                st.success("File caricati e corpus preparato con successo!")
            else:
                st.warning("Nessun dato valido trovato nei file caricati.")
                st.session_state.corpus_df = pd.DataFrame()
                
        except Exception as e:
            st.error(f"Errore durante il caricamento o la preparazione dei file: {e}\n\nTraceback:\n{traceback.format_exc()}")

    # Mostriamo lo stato del corpus
    if not st.session_state.corpus_df.empty:
        st.write(f"Corpus totale pronto con **{len(st.session_state.corpus_df)}** esempi.")
        if st.button("Mostra Anteprima Corpus"):
            st.dataframe(st.session_state.corpus_df.head(10))

    # Pulsante per avviare il fine-tuning (funzione fittizia)
    if st.button("Avvia Fine-Tuning del Modello", disabled=st.session_state.corpus_df.empty):
        with st.spinner("Addestramento del modello in corso..."):
            # Qui si integrerebbe la logica del fine-tuning (es. con transformers e peft)
            # Per ora, simuliamo il processo
            import time
            time.sleep(3)
            st.session_state.fine_tuning_state = {"status": "Completato!", "progress": 1.0, "current_step": "Modello pronto per la generazione."}
            st.success("Fine-Tuning completato con successo! Il modello è pronto.")

# =========================
# Scheda 'Genera Giudizi su File'
# =========================
with st.expander("📝 Genera Giudizi su File", expanded=True):
    st.header("Completa un File Excel")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello compilerà la colonna e potrai scaricare il file aggiornato.")
    
    # Widget per il caricamento del file e la selezione del foglio
    excel_file_input = st.file_uploader(
        "Carica file Excel da completare",
        type=["xlsx", "xls", "xlsm"]
    )
    
    if excel_file_input:
        # Carichiamo i nomi dei fogli del file
        try:
            xls = pd.ExcelFile(excel_file_input)
            st.session_state.excel_sheets = xls.sheet_names
            
            st.session_state.selected_sheet = st.selectbox(
                "Seleziona Foglio di Lavoro",
                options=st.session_state.excel_sheets,
                index=None
            )
            
            if st.session_state.selected_sheet:
                st.button("Avvia Generazione su File", key="process_excel", on_click=lambda: process_excel_for_judgments(excel_file_input, st.session_state.selected_sheet))
            
        except Exception as e:
            st.error(f"Errore nel caricamento dei fogli di lavoro: {e}")
            
    # Funzione per l'elaborazione del file (fittizia)
    def process_excel_for_judgments(file_data, selected_sheet):
        try:
            # Carichiamo il file in un DataFrame
            df_to_complete = pd.read_excel(file_data, sheet_name=selected_sheet)
            giudizio_col = find_giudizio_column(df_to_complete)

            if giudizio_col is None:
                st.warning("La colonna 'Giudizio' non è stata trovata. Impossibile procedere.")
                return

            # Simuliamo la generazione dei giudizi
            with st.spinner(f"Generazione giudizi per il foglio '{selected_sheet}' in corso..."):
                df_to_complete[giudizio_col] = df_to_complete.apply(
                    lambda row: f"Giudizio generato per la riga {row.name + 1}. (Simulato)", axis=1
                )

            st.session_state.generation_status = "Generazione completata con successo!"
            st.session_state.process_completed_file = df_to_complete

        except Exception as e:
            st.error(f"Errore durante la generazione: {e}\n\nTraceback:\n{traceback.format_exc()}")
            
    # Visualizzazione dello stato e del link per il download
    if st.session_state.generation_status:
        st.success(st.session_state.generation_status)
        if st.session_state.process_completed_file is not None:
            st.write("### Scarica il file completato")
            output_buffer = BytesIO()
            with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
            output_buffer.seek(0)
            
            st.download_button(
                label="Scarica il file aggiornato",
                data=output_buffer,
                file_name=f"giudizi_aggiornati.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
