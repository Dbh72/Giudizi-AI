# app.py

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import sys

# Importiamo il modulo che contiene la logica per leggere i file Excel.
from excel_reader import load_and_prepare_excel

# ==============================================================================
# SEZIONE 2: FUNZIONI PER LA GESTIONE DEI FILE
# ==============================================================================

def save_uploaded_file(uploaded_file):
    """Salva il file caricato dall'utente in una directory temporanea."""
    try:
        # Creiamo una cartella temporanea per i file caricati.
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
if 'excel_files' not in st.session_state:
    st.session_state.excel_files = []
if 'last_action_status' not in st.session_state:
    st.session_state.last_action_status = ""
if 'excel_content_history' not in st.session_state:
    st.session_state.excel_content_history = {}
if 'corpus' not in st.session_state:
    st.session_state.corpus = {}

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE (UI) DI STREAMLIT
# ==============================================================================

st.title("Generatore di Giudizi con IA 🎓")
st.markdown("Carica i tuoi file Excel per l'addestramento del modello. L'applicazione conserva lo stato di lavoro.")
st.write("---")

# Area per il caricamento dei file Excel
st.header("Carica File Excel (.xlsx, .xls, .xlsm)")
excel_uploaded_files = st.file_uploader(
    "Trascina qui i tuoi file Excel",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True,
    key="excel_uploader"
)

# Processiamo i file caricati se ce ne sono di nuovi
if excel_uploaded_files:
    for excel_file in excel_uploaded_files:
        if excel_file.name not in [f.name for f in st.session_state.excel_files]:
            st.session_state.excel_files.append(excel_file)
            st.session_state.last_action_status = f"File Excel {excel_file.name} caricato."

# Visualizziamo i file caricati
if st.session_state.excel_files:
    st.write("### File Excel Caricati")
    st.info(st.session_state.last_action_status)

    for i, file in enumerate(st.session_state.excel_files):
        with st.expander(f"**{file.name}**"):
            st.write(f"Dimensioni: {file.size / 1024:.2f} KB")

            process_button_key = f"process_excel_{i}"
            if st.button("Processa e Visualizza Contenuto", key=process_button_key):
                st.session_state.last_action_status = f"Elaborazione di {file.name}..."
                # Memorizziamo il nome del file da processare per il rerun
                st.session_state["file_to_process"] = file.name
                st.experimental_rerun()

            # Se il file è stato elaborato, mostriamo il suo contenuto
            if file.name in st.session_state.excel_content_history:
                st.write("Contenuto estratto (fogli):")
                for sheet_name, df in st.session_state.excel_content_history[file.name].items():
                    st.write(f"**Foglio:** `{sheet_name}`")
                    st.dataframe(df)

st.write("---")

# SEZIONE: LOGICA DI ELABORAZIONE (DOPO IL RERUN)
if "file_to_process" in st.session_state:
    file_to_process_name = st.session_state["file_to_process"]
    uploaded_file = next((f for f in st.session_state.excel_files if f.name == file_to_process_name), None)

    if uploaded_file:
        with st.spinner(f"Elaborazione di {uploaded_file.name}..."):
            try:
                # Chiamiamo la funzione per leggere l'Excel e prepararne i dati
                file_path = save_uploaded_file(uploaded_file)
                extracted_content = load_and_prepare_excel(file_path)

                if extracted_content:
                    st.session_state.excel_content_history[uploaded_file.name] = extracted_content
                    # Uniamo i dati di tutti i file per creare un corpus unificato
                    st.session_state.corpus[uploaded_file.name] = extracted_content
                    st.session_state.last_action_status = f"Contenuto di {uploaded_file.name} elaborato con successo!"
                else:
                     st.session_state.last_action_status = f"Impossibile elaborare il contenuto di {uploaded_file.name}."
            except Exception as e:
                st.session_state.last_action_status = f"Errore nell'elaborazione di {uploaded_file.name}: {e}"
            finally:
                os.remove(file_path)
                del st.session_state["file_to_process"]
                st.experimental_rerun()
