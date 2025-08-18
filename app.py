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
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}

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

# ==============================================================================
# SOTTO-SEZIONE: CARICAMENTO E PREPARAZIONE DEL CORPUS
# ==============================================================================
st.write("### Carica i File per il Corpus di Fine-Tuning")
st.write("Carica uno o più file Excel per costruire il corpus. I file devono contenere la colonna 'Giudizio'. Se ricarichi un file già presente, verrà aggiornato.")

uploaded_files = st.file_uploader(
    "Trascina e rilascia i file qui",
    type=["xlsx", "xls", "xlsm"],
    accept_multiple_files=True
)

if uploaded_files:
    # Aggiungi i nuovi file allo stato della sessione o aggiorna quelli esistenti
    for file in uploaded_files:
        st.session_state.uploaded_files_data[file.name] = file
    
    # Processa tutti i file memorizzati
    corpus_list = []
    
    # Usa st.status per mostrare lo stato di avanzamento in tempo reale
    with st.status("Elaborazione dei file...", expanded=True) as status:
        try:
            for file_name, file_data in st.session_state.uploaded_files_data.items():
                status.write(f"Elaborazione del file '{file_name}'...")
                
                temp_file_path = save_uploaded_file(file_data)
                if temp_file_path:
                    df_new = load_and_prepare_excel(temp_file_path)
                    
                    if not df_new.empty:
                        corpus_list.append(df_new)
                        status.write(f"File '{file_name}' elaborato con successo. Righe aggiunte: {len(df_new)}")
                    else:
                        status.write(f"ATTENZIONE: Il file '{file_name}' non contiene dati validi. Saltato.")
                    
                    # Rimuovi il file temporaneo
                    os.remove(temp_file_path)

            if corpus_list:
                st.session_state.corpus = pd.concat(corpus_list, ignore_index=True)
                total_rows = len(st.session_state.corpus)
                status.update(label=f"Corpus creato con {total_rows} righe. Operazione completata!", state="complete", expanded=False)
            else:
                st.session_state.corpus = pd.DataFrame()
                status.update(label="Nessun dato valido trovato nei file caricati.", state="error", expanded=True)

        except Exception as e:
            status.update(label=f"Errore durante l'elaborazione: {e}", state="error", expanded=True)
            st.error(f"Si è verificato un errore: {e}\n\nTraceback:\n{traceback.format_exc()}")
            

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
st.warning("Funzionalità di generazione non implementata in questo script. Se hai bisogno di aiuto con questa sezione, fammelo sapere.")

