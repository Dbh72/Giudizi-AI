# ==============================================================================
# File: app.py
# L'interfaccia utente principale per l'applicazione di generazione di giudizi.
# Utilizza Streamlit per creare un'interfaccia web interattiva.
# Questo file gestisce l'intero flusso di lavoro:
# 1. Caricamento dei file di addestramento e/o del modello fine-tunato.
# 2. Addestramento (fine-tuning) del modello.
# 3. Caricamento del file Excel da completare.
# 4. Generazione dei giudizi per il file caricato.
# 5. Download del file Excel completato.
# ==============================================================================

# SEZIONE 0: LIBRERIE NECESSARIE E CONFIGURAZIONE
# ==============================================================================
import streamlit as st
import pandas as pd
import os
import shutil
import warnings
import traceback
from datetime import datetime
from io import BytesIO
import json
import time
import requests
import re
import random 
import openpyxl
from collections import Counter
from openpyxl import Workbook, load_workbook

# ==============================================================================
# SEZIONE A: FUNZIONI PER LA GESTIONE DEI DATI EXCEL
# (dal tuo file excel_reader.py)
# ==============================================================================
class ExcelReader:
    """
    Classe per leggere e preparare dati da file Excel.
    """
    def find_header_row_and_columns(self, df):
        """
        Trova la riga di intestazione e le posizioni delle colonne 'input_text' e 'target_text'.
        """
        try:
            for i in range(min(50, len(df))):
                row = df.iloc[i]
                row_str = " ".join(row.astype(str).tolist()).lower()
                # Use a regular expression to match "descrizione" or "description"
                if re.search(r'\b(descrizione|description)\b', row_str) and re.search(r'\b(giudizio|judgment)\b', row_str):
                    header_row_index = i
                    headers = df.iloc[header_row_index].astype(str).tolist()
                    description_col_name = next((h for h in headers if re.search(r'\b(descrizione|description)\b', h.lower())), None)
                    judgment_col_name = next((h for h in headers if re.search(r'\b(giudizio|judgment)\b', h.lower())), None)
                    
                    if description_col_name and judgment_col_name:
                        return header_row_index, description_col_name, judgment_col_name
            return None, None, None
        except Exception as e:
            raise Exception(f"Errore nella ricerca dell'intestazione: {e}")

    def read_and_prepare_data_from_excel(self, file_object, sheet_names, progress_container):
        """
        Legge i dati dai fogli specificati, prepara un dataframe con 'input_text' e 'target_text'.
        """
        try:
            corpus_list = []
            
            for sheet in sheet_names:
                progress_container(f"Lettura del foglio: '{sheet}'...", "info")
                file_object.seek(0)
                try:
                    df = pd.read_excel(file_object, sheet_name=sheet, header=None)
                except Exception as e:
                    progress_container(f"Impossibile leggere il foglio '{sheet}'. Ignorato. Errore: {e}", "warning")
                    continue
                
                header_row_index, description_col, judgment_col = self.find_header_row_and_columns(df)
                
                if header_row_index is None:
                    progress_container(f"Attenzione: Intestazione non trovata nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                df.columns = df.iloc[header_row_index]
                df = df.iloc[header_row_index + 1:].reset_index(drop=True)
                
                if description_col not in df.columns or judgment_col not in df.columns:
                    progress_container(f"Attenzione: Colonne 'Descrizione' o 'Giudizio' non trovate nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                df_subset = df[[description_col, judgment_col]].rename(columns={description_col: 'input_text', judgment_col: 'target_text'})
                
                data_for_dataset = []
                for _, row in df_subset.iterrows():
                    input_text = str(row['input_text']).strip()
                    target_text = str(row['target_text']).strip()
                    
                    if pd.notna(input_text) and pd.notna(target_text) and input_text and target_text != "nan":
                        data_for_dataset.append({
                            'input_text': input_text,
                            'target_text': target_text
                        })

                if not data_for_dataset:
                    progress_container(f"Attenzione: Nessun dato valido trovato nel foglio '{sheet}'. Saltato.", "warning")
                    continue
                
                corpus_list.extend(data_for_dataset)
            
            if not corpus_list:
                progress_container("Nessun dato valido trovato in tutti i fogli del file.", "error")
                return pd.DataFrame()
            
            return pd.DataFrame(corpus_list)

        except Exception as e:
            progress_container(f"Errore nella lettura del file: {e}", "error")
            progress_container(f"Traceback: {traceback.format_exc()}", "error")
            return pd.DataFrame()

    def get_excel_sheet_names(self, file_object):
        """
        Restituisce una lista con i nomi di tutti i fogli di lavoro in un file Excel.
        """
        try:
            file_object.seek(0)
            workbook = openpyxl.load_workbook(file_object, read_only=True)
            sheet_names = workbook.sheetnames
            workbook.close()
            return sheet_names
        except Exception as e:
            progress_container(f"Errore nella lettura dei nomi dei fogli: {e}", "error")
            return []

    def get_file_type(self, file_object):
        """
        Determina il tipo di file Excel in base al contenuto.
        """
        file_object.seek(0)
        signature = file_object.read(8)
        file_object.seek(0)
        
        if signature == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
            return 'xls'
        elif signature[:4] == b'PK\x03\x04':
            return 'xlsx'
        else:
            return None

# ==============================================================================
# SEZIONE B: FUNZIONI PER L'ADDESTRAMENTO E LA GENERAZIONE
# (mock)
# ==============================================================================

class MockModel:
    def __init__(self):
        pass

class MockTokenizer:
    def __init__(self):
        pass

class MockModelTrainer:
    def train_model(self, corpus_df, progress_container):
        progress_container("MOCK: Addestramento del modello in corso...", "info")
        for i in range(1, 6):
            time.sleep(1)
            progress_container(f"MOCK: Addestramento - Fase {i}/5...", "info")
        progress_container("MOCK: Addestramento completato.", "success")
        return MockModel(), MockTokenizer()

    def load_fine_tuned_model(self, progress_container):
        progress_container("MOCK: Caricamento del modello esistente...", "info")
        time.sleep(1)
        # Simula il caso in cui il modello non è sempre disponibile
        if random.choice([True, False]):
             progress_container("MOCK: Modello esistente caricato.", "success")
             return MockModel(), MockTokenizer()
        else:
            progress_container("MOCK: Nessun modello esistente trovato.", "error")
            return None, None

    def delete_model(self, progress_container):
        progress_container("MOCK: Eliminazione del modello...", "info")
        time.sleep(1)
        progress_container("MOCK: Modello eliminato.", "success")

class MockJudgmentGenerator:
    def generate_judgments(self, df, model, tokenizer, sheet_name, progress_container):
        progress_container("MOCK: Generazione dei giudizi in corso...", "info")
        time.sleep(2)
        # Simula l'aggiunta di una colonna "Giudizio_Generato"
        df['Giudizio_Generato'] = df['input_text'].apply(lambda x: f"Giudizio generato per: {x}")
        return df

class MockCorpusBuilder:
    def build_or_update_corpus(self, new_df, progress_container):
        progress_container("MOCK: Aggiornamento del corpus...", "info")
        # Simula l'unione di dati
        corpus = st.session_state.corpus_df
        updated_corpus = pd.concat([corpus, new_df], ignore_index=True)
        return updated_corpus.drop_duplicates(subset=['input_text'], keep='first')

    def delete_corpus(self, progress_container):
        progress_container("MOCK: Eliminazione del corpus...", "info")
        time.sleep(1)
        progress_container("MOCK: Corpus eliminato.", "success")
        
# Istanzio le classi
er = ExcelReader()
mt = MockModelTrainer()
jg = MockJudgmentGenerator()
cb = MockCorpusBuilder()

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# Definizione della variabile dell'API Key per l'uso con le API di Gemini
GEMINI_API_KEY = ""

# ==============================================================================
# SEZIONE 1: FUNZIONI AUSILIARIE
# ==============================================================================

def progress_container(message, type):
    """
    Gestisce l'aggiornamento del placeholder di stato con un messaggio.
    """
    if "status_placeholder" not in st.session_state:
        st.session_state.status_placeholder = st.empty()
    
    if type == "info":
        st.session_state.status_placeholder.info(message)
    elif type == "success":
        st.session_state.status_placeholder.success(message)
    elif type == "warning":
        st.session_state.status_placeholder.warning(message)
    elif type == "error":
        st.session_state.status_placeholder.error(message)

def reset_project_state():
    """
    Resetta tutti i file del progetto (corpus, modello) e lo stato di sessione.
    """
    try:
        # Elimina il corpus (mock)
        cb.delete_corpus(lambda msg, type: progress_container(msg, type))
        # Elimina il modello (mock)
        mt.delete_model(lambda msg, type: progress_container(msg, type))
        progress_container("Progetto resettato. Tutti i dati sono stati eliminati.", "success")
        
        # Reset dello stato di sessione
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
    except Exception as e:
        progress_container(f"Errore durante il reset del progetto: {e}", "error")
        st.error(f"Errore: {traceback.format_exc()}")
        
    st.experimental_rerun()
    
def get_session_id():
    """
    Genera un ID di sessione per distinguere i file in un ambiente multi-utente.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    return st.session_state.session_id

# ==============================================================================
# SEZIONE 2: INTERFACCIA UTENTE E LOGICA PRINCIPALE
# ==============================================================================

# Impostazioni della pagina
st.set_page_config(
    page_title="Giudizi-AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inizializzazione di Streamlit session state per mantenere lo stato dell'app
# Inizializzazione Condizionale per evitare errori all'avvio
if 'corpus_df' not in st.session_state:
    st.session_state.corpus_df = pd.DataFrame()
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'selected_sheet_trainer' not in st.session_state:
    st.session_state.selected_sheet_trainer = None
if 'uploaded_process_file' not in st.session_state:
    st.session_state.uploaded_process_file = None
if 'selected_sheet_process' not in st.session_state:
    st.session_state.selected_sheet_process = None
if 'process_completed_file' not in st.session_state:
    st.session_state.process_completed_file = None
if 'status_placeholder' not in st.session_state:
    st.session_state.status_placeholder = st.empty()


# Titolo e descrizione dell'app
st.title("🤖 Giudizi-AI: Generatore di Giudizi 🤖")
st.markdown("---")
st.markdown("Questa applicazione ti aiuta a generare automaticamente giudizi per le tue valutazioni. Segui i passaggi di seguito.")

# Layout in colonne per la sidebar e il contenuto principale
main_col, sidebar_col = st.columns([3, 1])

with sidebar_col:
    st.markdown("### Impostazioni e Stato")
    
    st.info("Utilizza questa sidebar per resettare l'applicazione o per informazioni di stato.")
    
    # Bottone per caricare il modello
    if st.button("Carica Modello Esistente"):
        try:
            model, tokenizer = mt.load_fine_tuned_model(lambda msg, type: progress_container(msg, type))
            if model and tokenizer:
                st.session_state.model_ready = True
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                progress_container("Modello caricato con successo.", "success")
            else:
                st.session_state.model_ready = False
                progress_container("Nessun modello trovato. Devi addestrarne uno.", "error")
        except Exception as e:
            progress_container(f"Errore durante il caricamento del modello: {e}", "error")
            st.error(f"Errore: {traceback.format_exc()}")
            
    # Bottone per resettare lo stato del progetto
    if st.button("Resetta tutto il progetto"):
        reset_project_state()
        
    st.markdown("---")
    st.write("### Stato Attuale")
    if st.session_state.model_ready:
        st.success("✅ Modello caricato e pronto!")
    else:
        st.error("❌ Nessun modello caricato.")
        
    st.write(f"Righe nel corpus: {len(st.session_state.corpus_df)}")
    
    # Mostra l'ID della sessione
    st.text(f"ID Sessione: {get_session_id()}")

# ==============================================================================
# SEZIONE 1: ADDESTRAMENTO DEL MODELLO
# ==============================================================================
with main_col:
    st.header("1. Addestramento del Modello 🧠")
    st.info("Carica uno o più file Excel per creare o aggiornare il corpus di addestramento. Le colonne 'Descrizione' e 'Giudizio' verranno utilizzate per il fine-tuning del modello.")
    
    # Placeholder per i messaggi di stato
    st.session_state.status_placeholder = st.empty()
    
    # Caricamento dei file per l'addestramento
    uploaded_files_trainer = st.file_uploader(
        "Carica i file di addestramento (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        accept_multiple_files=True,
        key="uploader_trainer"
    )
    
    # Aggiorno lo stato dei file caricati
    if uploaded_files_trainer:
        st.session_state.uploaded_files = uploaded_files_trainer
        try:
            first_file = uploaded_files_trainer[0]
            sheet_names = er.get_excel_sheet_names(first_file)
            st.session_state.selected_sheet_trainer = st.selectbox(
                "Seleziona il foglio di lavoro da utilizzare per l'addestramento:",
                sheet_names,
                key="sheet_select_trainer"
            )
        except Exception as e:
            progress_container(f"Errore nella lettura dei fogli del file: {e}", "error")
            st.error(f"Errore: {traceback.format_exc()}")
        
    if st.button("Avvia Addestramento"):
        if st.session_state.uploaded_files:
            try:
                progress_container("Lettura dei file e preparazione del corpus...", "info")
                new_df = pd.DataFrame()
                for file in st.session_state.uploaded_files:
                    if st.session_state.selected_sheet_trainer:
                        df_temp = er.read_and_prepare_data_from_excel(file, [st.session_state.selected_sheet_trainer], lambda msg, type: progress_container(msg, type))
                        new_df = pd.concat([new_df, df_temp], ignore_index=True)
                    else:
                        progress_container("Seleziona un foglio di lavoro.", "warning")
                        continue

                st.session_state.corpus_df = cb.build_or_update_corpus(new_df, lambda msg, type: progress_container(msg, type))
                
                if not st.session_state.corpus_df.empty:
                    st.session_state.model, st.session_state.tokenizer = mt.train_model(st.session_state.corpus_df, lambda msg, type: progress_container(msg, type))
                    if st.session_state.model and st.session_state.tokenizer:
                        st.session_state.model_ready = True
                        progress_container("Addestramento completato con successo!", "success")
                        st.balloons()
                    else:
                        st.session_state.model_ready = False
                        progress_container("Errore durante l'addestramento. Controlla i dati.", "error")
                else:
                    progress_container("Corpus vuoto. Impossibile avviare l'addestramento.", "error")
            except Exception as e:
                progress_container(f"Errore durante il processo di addestramento: {e}", "error")
                st.error(f"Errore: {traceback.format_exc()}")
        else:
            st.warning("Per addestrare il modello, devi prima caricare almeno un file e selezionare un foglio.")

st.markdown("---")

# ==============================================================================
# SEZIONE 2: GENERAZIONE DEI GIUDIZI
# ==============================================================================
with main_col:
    st.header("2. Generazione dei Giudizi 🤖")
    st.info("Carica il file Excel che vuoi completare. L'applicazione genererà i giudizi per le righe mancanti.")
    
    # Caricamento del file per la generazione
    uploaded_process_file = st.file_uploader(
        "Carica il file da completare (.xlsx, .xls, .xlsm)",
        type=["xlsx", "xls", "xlsm"],
        key="uploader_process"
    )
    
    if uploaded_process_file:
        st.session_state.uploaded_process_file = uploaded_process_file
        try:
            sheet_names = er.get_excel_sheet_names(uploaded_process_file)
            st.session_state.selected_sheet_process = st.selectbox(
                "Seleziona il foglio di lavoro da completare:",
                sheet_names,
                key="sheet_select_process"
            )
        except Exception as e:
            progress_container(f"Errore nella lettura dei fogli del file: {e}", "error")
            st.error(f"Errore: {traceback.format_exc()}")
        
    if st.button("Genera Giudizi"):
        if st.session_state.model_ready:
            if st.session_state.uploaded_process_file and st.session_state.selected_sheet_process:
                try:
                    progress_container("Lettura del file...", "info")
                    df = er.read_and_prepare_data_from_excel(st.session_state.uploaded_process_file, [st.session_state.selected_sheet_process], lambda msg, type: progress_container(msg, type))
                    
                    if not df.empty:
                        progress_container("Generazione dei giudizi in corso...", "info")
                        df_with_judgments = jg.generate_judgments(df, st.session_state.model, st.session_state.tokenizer, st.session_state.selected_sheet_process, lambda msg, type: progress_container(msg, type))
                        
                        st.session_state.process_completed_file = df_with_judgments
                        st.session_state.selected_sheet = st.session_state.selected_sheet_process
                        progress_container("Generazione completata!", "success")
                    else:
                        progress_container("Il file caricato è vuoto o non contiene dati validi.", "error")
                except Exception as e:
                    progress_container(f"Errore nella generazione dei giudizi: {e}", "error")
                    st.error(f"Errore: {traceback.format_exc()}")
            else:
                st.warning("Per generare i giudizi, devi prima caricare un file nella sezione '2. Generazione dei Giudizi'.")
        else:
            st.warning("Per generare i giudizi, devi prima addestrare un modello nella sezione '1. Addestramento del Modello'.")

st.markdown("---")

# ==============================================================================
# SEZIONE 3: VISUALIZZAZIONE RISULTATI E DOWNLOAD
# ==============================================================================
with main_col:
    st.header("3. Stato e Download")

    if st.session_state.process_completed_file is not None:
        st.write("### Scarica il file completato")
        st.write("Di seguito, puoi vedere l'anteprima del file con i giudizi generati e scaricare il documento completo.")
        
        st.dataframe(st.session_state.process_completed_file)
        
        output_buffer = BytesIO()
        with pd.ExcelWriter(output_buffer, engine='openpyxl') as writer:
            st.session_state.process_completed_file.to_excel(writer, index=False, sheet_name=st.session_state.selected_sheet)
        output_buffer.seek(0)
        
        st.download_button(
            label="Scarica il file aggiornato",
            data=output_buffer,
            file_name=f"Giudizi_Generati_{st.session_state.selected_sheet}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("I risultati appariranno qui una volta che avrai generato i giudizi.")
