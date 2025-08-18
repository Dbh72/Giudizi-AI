# ==============================================================================
# File: app.py
# Orchestratore principale con funzionalità unificate Streamlit
# ==============================================================================

import streamlit as st
import pandas as pd
import os
import shutil
import traceback
from io import BytesIO

# Importiamo i moduli con la logica specifica.
# Assicurati che i file 'excel_reader.py', 'model_trainer.py', e 'judgment_generator.py'
# si trovino nella stessa directory.
from excel_reader import load_and_prepare_excel, find_giudizio_column
# from model_trainer import fine_tune_model # Abilita quando il modulo è pronto
# from judgment_generator import generate_judgments_for_excel # Abilita quando il modulo è pronto

# ==============================================================================
# SEZIONE 1: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

# Impostiamo il titolo della pagina e l'icona per l'app Streamlit.
st.set_page_config(
    page_title="Generatore di Giudizi con IA",
    page_icon="🤖",
    layout="wide"
)

# Directory dove verrà salvato il modello addestrato.
OUTPUT_DIR = "modello_finetunato"

# Inizializziamo le variabili di stato della sessione per mantenere i dati tra le interazioni.
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'fine_tuning_status' not in st.session_state:
    st.session_state.fine_tuning_status = ""
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = ""
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'model_saved_path' not in st.session_state:
    st.session_state.model_saved_path = None
if 'selected_sheet' not in st.session_state:
    st.session_state.selected_sheet = None


# ==============================================================================
# SEZIONE 2: FUNZIONI PER L'INTERFACCIA UTENTE
# ==============================================================================

def update_corpus_from_files(uploaded_files):
    """
    Legge i file Excel caricati, li processa e costruisce il corpus di addestramento.
    """
    if not uploaded_files:
        st.session_state.uploaded_files_data = {}
        st.session_state.corpus_df = pd.DataFrame()
        st.info("Nessun file caricato o i file sono stati rimossi.")
        return

    st.info("Elaborazione dei file in corso... Attendere.")
    all_data = []
    
    # Aggiorna il dizionario di stato con i nuovi file
    for file in uploaded_files:
        st.session_state.uploaded_files_data[file.name] = file.getvalue()

    # Ricostruisce il corpus totale da zero usando tutti i file correnti
    for file_name, file_data in st.session_state.uploaded_files_data.items():
        try:
            # Chiama la funzione di lettura del modulo 'excel_reader'
            df_file = load_and_prepare_excel(BytesIO(file_data), file_name)
            if not df_file.empty:
                all_data.append(df_file)
        except Exception as e:
            st.error(f"Errore nella lettura del file '{file_name}': {e}")
            st.error(traceback.format_exc())

    if all_data:
        st.session_state.corpus_df = pd.concat(all_data, ignore_index=True)
        st.success(f"Corpus creato con successo! Trovate {len(st.session_state.corpus_df)} righe valide per l'addestramento.")
    else:
        st.warning("Nessun dato valido trovato in tutti i file caricati.")
        st.session_state.corpus_df = pd.DataFrame()


def handle_fine_tuning():
    """
    Gestisce il processo di fine-tuning del modello.
    """
    if st.button("Avvia Fine-Tuning", type="primary"):
        if not st.session_state.corpus_df.empty:
            # Controlla se il modulo model_trainer è disponibile
            # if 'fine_tune_model' in globals():
            #     with st.spinner("Addestramento del modello in corso..."):
            #         # Simula l'addestramento e il salvataggio del modello
            #         st.session_state.fine_tuning_status = "Simulazione: Modello addestrato e salvato."
            #         # st.session_state.model_saved_path = fine_tune_model(st.session_state.corpus_df)
            #         st.session_state.model_saved_path = OUTPUT_DIR # Simulazione
            # else:
            #     st.error("Modulo 'model_trainer' non trovato. Assicurati che il file esista e sia importato.")
            # st.write("Nota: L'addestramento del modello è una simulazione. Per il vero addestramento, implementa la logica nel modulo `model_trainer.py`.")
            st.info("L'addestramento del modello richiede un ambiente specifico (GPU, librerie) non disponibile in questo contesto. Per ora, simuliamo il successo dell'operazione.")
            st.session_state.fine_tuning_status = "Simulazione: Addestramento completato. Modello salvato."
            st.session_state.model_saved_path = OUTPUT_DIR # Simulazione
        else:
            st.warning("Per avviare l'addestramento, devi prima caricare dei file e creare il corpus.")


def handle_judgment_generation(file_to_complete, selected_sheet):
    """
    Gestisce la generazione dei giudizi per un file Excel.
    """
    if st.button("Avvia Generazione su File", type="primary", disabled=(file_to_complete is None or not selected_sheet)):
        if file_to_complete is None or not selected_sheet:
            st.warning("Per favore, carica un file Excel e seleziona un foglio di lavoro.")
            return

        st.session_state.generation_status = ""
        st.session_state.process_completed_file = None

        try:
            # Chiama la funzione di lettura del modulo 'excel_reader' per preparare il file
            df_to_complete, header_row = load_and_prepare_excel(file_to_complete, file_to_complete.name, selected_sheet, get_header_row=True)
            if df_to_complete.empty:
                st.warning(f"Il foglio '{selected_sheet}' non contiene dati validi. Operazione annullata.")
                return

            giudizio_col = find_giudizio_column(df_to_complete)
            if not giudizio_col:
                st.error("Colonna 'Giudizio' non trovata nel foglio selezionato. Assicurati che sia presente.")
                return

            # Controlla se il modulo judgment_generator è disponibile
            # if 'generate_judgments_for_excel' in globals():
            #     with st.spinner(f"Generazione giudizi per il foglio '{selected_sheet}' in corso..."):
            #         # Simula il processo di generazione e aggiorna lo stato
            #         # updated_df = generate_judgments_for_excel(df_to_complete, giudizio_col)
            #         updated_df = df_to_complete.copy()
            #         updated_df[giudizio_col] = updated_df.apply(lambda row: f"Giudizio generato per la riga {row.name + 1}. (Simulato)", axis=1)
            #         st.session_state.generation_status = "Generazione completata con successo!"
            #         st.session_state.process_completed_file = updated_df
            # else:
            #     st.error("Modulo 'judgment_generator' non trovato. Assicurati che il file esista e sia importato.")
            # st.write("Nota: La generazione dei giudizi è una simulazione. Per il vero processo, implementa la logica nel modulo `judgment_generator.py`.")
            st.info("La generazione dei giudizi richiede un modello addestrato. Per ora, simuliamo il successo dell'operazione.")
            updated_df = df_to_complete.copy()
            updated_df[giudizio_col] = updated_df.apply(lambda row: f"Giudizio generato per la riga {row.name + 1}. (Simulato)", axis=1)
            st.session_state.generation_status = "Simulazione: Generazione completata con successo!"
            st.session_state.process_completed_file = updated_df

        except Exception as e:
            st.error(f"Errore durante la generazione: {e}\n\nTraceback:\n{traceback.format_exc()}")
            st.session_state.generation_status = f"Errore: {e}"


def get_excel_sheet_names(uploaded_file):
    """
    Estrae i nomi dei fogli di lavoro da un file Excel caricato.
    """
    if uploaded_file is not None:
        try:
            xls = pd.ExcelFile(uploaded_file)
            return xls.sheet_names
        except Exception:
            return ["Errore nella lettura dei fogli"]
    return []

# ==============================================================================
# SEZIONE 3: LAYOUT DELL'APPLICAZIONE (TAB)
# ==============================================================================

st.title("Generatore di Giudizi con IA")
st.markdown("Benvenuto! Usa questa applicazione per addestrare un modello di IA a generare giudizi a partire dai tuoi dati di valutazione.")

tab1, tab2 = st.tabs(["**Addestramento (Fine-Tuning)**", "**Generazione di Giudizi**"])

with tab1:
    st.header("1. Addestramento del Modello")
    st.markdown("Carica i file Excel con i dati che userai per addestrare il modello. Le righe vuote verranno ignorate.")
    
    fine_tune_file_input = st.file_uploader(
        "Carica file Excel per l'addestramento",
        type=['xlsx', 'xls', 'xlsm'],
        accept_multiple_files=True,
        on_change=lambda: update_corpus_from_files(st.session_state.fine_tune_file_input)
    )

    if fine_tune_file_input:
        st.session_state.fine_tune_file_input = fine_tune_file_input
        # Mostra il riepilogo del corpus
        st.write("### Riepilogo del Corpus di Addestramento")
        if not st.session_state.corpus_df.empty:
            st.dataframe(st.session_state.corpus_df.head(10))
            st.write(f"Numero totale di esempi validi: **{len(st.session_state.corpus_df)}**")
        else:
            st.warning("Corpus non ancora pronto. Carica i file per iniziare.")
        
        # Pulsante per avviare il fine-tuning
        handle_fine_tuning()
        
        # Mostra lo stato del fine-tuning
        if st.session_state.fine_tuning_status:
            st.info(st.session_state.fine_tuning_status)

        # Pulsante di download del modello
        if st.session_state.model_saved_path:
            with open("modello_finetunato.zip", "rb") as fp:
                 st.download_button(
                    label="Scarica il Modello Addestrato",
                    data=fp,
                    file_name="modello_finetunato.zip",
                    mime="application/zip",
                    help="Puoi scaricare il modello per riutilizzarlo in seguito."
                )

with tab2:
    st.header("2. Generazione di Giudizi su File")
    st.markdown("Carica un file Excel con la colonna 'Giudizio' vuota. Il modello la compilerà e potrai scaricare il file aggiornato.")

    file_to_complete = st.file_uploader(
        "Carica file Excel da completare",
        type=['xlsx', 'xls', 'xlsm']
    )

    if file_to_complete:
        st.session_state.file_to_complete = file_to_complete
        sheet_names = get_excel_sheet_names(file_to_complete)
        
        selected_sheet = st.selectbox(
            "Seleziona il Foglio di Lavoro",
            options=sheet_names,
            index=0
        )
        st.session_state.selected_sheet = selected_sheet

        handle_judgment_generation(file_to_complete, selected_sheet)

        # Mostra lo stato dell'operazione e il link per il download
        if st.session_state.generation_status:
            st.info(st.session_state.generation_status)
            if st.session_state.process_completed_file is not None:
                st.write("### Scarica il file completato")
                output_buffer = BytesIO()
                with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
                    st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
                output_buffer.seek(0)
                
                st.download_button(
                    label="Scarica il file aggiornato",
                    data=output_buffer,
                    file_name=f"giudizi_aggiornati_{st.session_state.selected_sheet}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

