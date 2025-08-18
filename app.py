# ==============================================================================
# File: app.py
# Orchestratore principale con interfaccia utente Streamlit.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo le librerie essenziali per l'applicazione.
import streamlit as st
import pandas as pd
import os
from io import BytesIO
import traceback
import warnings
from datetime import datetime

# Importiamo i moduli con la logica per la preparazione dei dati, l'addestramento e la generazione.
from excel_reader import load_and_prepare_excel, find_giudizio_column
from model_trainer import fine_tune_model
from judgment_generator import generate_judgments_for_excel, load_trained_model

# Ignoriamo i FutureWarning per mantenere la console pulita
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

# Impostiamo il titolo della pagina e l'icona per l'app Streamlit.
st.set_page_config(
    page_title="Generatore di Giudizi con IA",
    page_icon="🤖",
    layout="wide"
)

# Inizializziamo le variabili di stato della sessione per mantenere i dati.
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'fine_tuning_status' not in st.session_state:
    st.session_state.fine_tuning_status = {"status": "In attesa", "model_path": None, "last_checkpoint": None}
if 'generation_status' not in st.session_state:
    st.session_state.generation_status = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'status_messages' not in st.session_state:
    st.session_state.status_messages = []
if 'progress_bar' not in st.session_state:
    st.session_state.progress_bar = None
if 'model_load_status' not in st.session_state:
    st.session_state.model_load_status = "In attesa"

# Funzione per aggiungere messaggi di stato
def add_status_message(message, level="info"):
    """Aggiunge un messaggio di stato alla lista."""
    st.session_state.status_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
    st.session_state.status_messages = st.session_state.status_messages[-20:]  # Mantieni solo gli ultimi 20 messaggi
    if level == "info":
        st.info(message)
    elif level == "success":
        st.success(message)
    elif level == "error":
        st.error(message)

# Inizializza la funzione di callback nello stato della sessione
st.session_state.add_status_message = add_status_message


# ==============================================================================
# SEZIONE 3: INTERFACCIA UTENTE
# ==============================================================================
st.title("🤖 Generatore di Giudizi con IA")
st.markdown("---")

# SEZIONE 3.1: Addestramento del Modello
st.subheader("Addestramento del Modello (Fine-Tuning)")
with st.expander("Carica file per l'addestramento e avvia il fine-tuning"):
    st.markdown("""
        Carica uno o più file Excel (.xlsx, .xlsm) che contengano una colonna di testo
        chiamata "Giudizio". Le altre colonne verranno utilizzate come dati di input
        per addestrare un modello di intelligenza artificiale a generare giudizi
        simili a quelli forniti.
    """)
    uploaded_files = st.file_uploader(
        "Carica uno o più file Excel",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Prepara i dati e avvia il fine-tuning"):
            with st.spinner("Preparazione dei dati per il fine-tuning..."):
                try:
                    # Chiamata al modulo per la preparazione dei dati
                    corpus_df = load_and_prepare_excel(uploaded_files)
                    
                    if not corpus_df.empty:
                        st.session_state.corpus_df = corpus_df
                        st.session_state.add_status_message(
                            f"Corpus totale creato con {len(st.session_state.corpus_df)} righe."
                        )

                        # Chiamata al modulo per il fine-tuning
                        st.session_state.add_status_message("Avvio del fine-tuning. Il processo potrebbe richiedere del tempo.")
                        model_path = fine_tune_model(st.session_state.corpus_df)
                        
                        if model_path:
                            st.session_state.fine_tuning_status = {
                                "status": "Completato",
                                "model_path": model_path
                            }
                            st.session_state.add_status_message("Addestramento completato con successo!", level="success")
                        else:
                            st.session_state.add_status_message("Addestramento fallito. Controlla i log per i dettagli.", level="error")

                    else:
                        st.session_state.add_status_message("Nessun dato valido trovato per il fine-tuning.", level="error")
                except Exception as e:
                    st.error(f"Errore durante la preparazione dei dati: {e}\n\nTraceback:\n{traceback.format_exc()}")


# SEZIONE 3.2: Generazione dei Giudizi
st.markdown("---")
st.subheader("Generazione di Giudizi su File Excel")
with st.expander("Carica un file da completare e avvia la generazione"):
    st.markdown("""
        Carica un file Excel con una colonna "Giudizio" vuota. Il modello addestrato
        compilerrà la colonna e potrai scaricare il file aggiornato.
    """)
    uploaded_file_to_complete = st.file_uploader(
        "Carica file Excel da completare",
        type=["xlsx", "xlsm"],
        accept_multiple_files=False,
        key="file_to_complete"
    )

    if uploaded_file_to_complete:
        try:
            file_data = {
                "name": uploaded_file_to_complete.name,
                "data": uploaded_file_to_complete.getvalue()
            }
            
            # Identifica i fogli di lavoro e la colonna 'Giudizio'
            excel_file = pd.ExcelFile(BytesIO(file_data["data"]))
            sheet_names = excel_file.sheet_names
            
            # Seleziona il foglio
            selected_sheet = st.selectbox(
                "Seleziona un foglio di lavoro",
                sheet_names,
                key="selected_sheet"
            )

            # Leggi il DataFrame del foglio selezionato
            df_to_complete = pd.read_excel(BytesIO(file_data["data"]), sheet_name=selected_sheet)
            giudizio_col = find_giudizio_column(df_to_complete)

            if not giudizio_col:
                st.warning(f"La colonna 'Giudizio' non è stata trovata nel foglio '{selected_sheet}'.")
            
            st.session_state.uploaded_files_data['file_to_complete'] = {
                'df': df_to_complete,
                'sheet': selected_sheet,
                'giudizio_col': giudizio_col
            }
            
            if st.button("Avvia Generazione su File", key="process_excel_button"):
                if st.session_state.fine_tuning_status["model_path"]:
                    try:
                        st.session_state.add_status_message("Avvio della generazione dei giudizi...")
                        
                        # Carica il modello e il tokenizer addestrati
                        st.session_state.add_status_message("Caricamento del modello addestrato...")
                        model, tokenizer = load_trained_model(st.session_state.fine_tuning_status["model_path"])
                        st.session_state.model_load_status = "Caricato con successo"
                        
                        if model and tokenizer:
                            st.session_state.add_status_message("Modello caricato. Inizio la generazione.")
                            completed_df = generate_judgments_for_excel(
                                model, tokenizer, df_to_complete, giudizio_col, selected_sheet,
                                st.session_state.fine_tuning_status["model_path"]
                            )
                            
                            st.session_state.process_completed_file = completed_df
                            st.success("Generazione completata con successo!")
                            st.balloons()
                        else:
                            st.error("Errore nel caricamento del modello. La generazione non può proseguire.")
                            st.session_state.model_load_status = "Errore di caricamento"
                    
                    except Exception as e:
                        st.error(f"Errore durante l'operazione: {e}\n\nTraceback:\n{traceback.format_exc()}")
                else:
                    st.warning("Per avviare la generazione, devi prima completare l'addestramento del modello.")

        except Exception as e:
            st.error(f"Errore nel caricamento del file Excel: {e}\n\nTraceback:\n{traceback.format_exc()}")


# SEZIONE 4: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
st.markdown("---")
st.subheader("Stato dell'Operazione e Download")

if st.session_state.status_messages:
    status_container = st.container()
    with status_container:
        for message in st.session_state.status_messages:
            st.info(message)
else:
    st.info("In attesa di un'operazione...")

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
        file_name=f"Giudizi_Generati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
