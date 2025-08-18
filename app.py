# app.py - Orchestratore principale per la creazione del Corpus

# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importa le librerie essenziali per l'applicazione.
# streamlit per la creazione dell'interfaccia utente web.
# pandas per la manipolazione dei dati in formato DataFrame.
# openpyxl per la lettura e scrittura di file Excel.
# re per le espressioni regolari per trovare la colonna 'Giudizio'.
# io.BytesIO per gestire i file in memoria.
import streamlit as st
import pandas as pd
import openpyxl
import re
from io import BytesIO
import traceback

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE DELLA PAGINA E GESTIONE DELLO STATO
# ==============================================================================

# Imposta il titolo della pagina e l'icona per l'app Streamlit.
st.set_page_config(
    page_title="Costruttore di Corpus",
    page_icon="📚",
    layout="wide"
)

# Inizializza le variabili di stato della sessione per mantenere i dati tra le interazioni.
if 'corpus' not in st.session_state:
    st.session_state.corpus = pd.DataFrame()
if 'uploaded_files_data' not in st.session_state:
    st.session_state.uploaded_files_data = {}

# ==============================================================================
# SEZIONE 3: FUNZIONI PER LA PREPARAZIONE DEI DATI DA FILE EXCEL
# ==============================================================================

def find_giudizio_column(df):
    """
    Trova la colonna 'Giudizio' nel DataFrame, cercando in modo case-insensitive
    in tutte le intestazioni.

    Args:
        df (pd.DataFrame): Il DataFrame del foglio da analizzare.

    Returns:
        str: Il nome della colonna 'Giudizio' o None se non trovata.
    """
    # Cerca la parola 'giudizio' in modo case-insensitive tra le colonne.
    for col in df.columns:
        if isinstance(col, str) and re.search(r'giudizio', str(col), re.IGNORECASE):
            return col
    return None

def find_header_row(file_content, sheet_name):
    """
    Scansiona le prime righe di un foglio di lavoro per identificare la riga
    dell'intestazione che contiene la colonna 'Giudizio'.

    Args:
        file_content (BytesIO): Il contenuto del file Excel in memoria.
        sheet_name (str): Il nome del foglio di lavoro.

    Returns:
        int: L'indice della riga dell'intestazione (0-based) o None se non trovata.
    """
    # Prova a leggere le prime 50 righe per trovare l'intestazione
    for header_row in range(50):
        try:
            # Usiamo BytesIO per passare il contenuto in memoria a pandas
            df_check = pd.read_excel(file_content, sheet_name=sheet_name, header=header_row)
            if find_giudizio_column(df_check):
                return header_row
        except Exception:
            file_content.seek(0)
            continue
    return None

def load_and_prepare_excel(file_content):
    """
    Carica un file Excel da un buffer di memoria, prepara i dati per il fine-tuning
    e li restituisce come un DataFrame di Pandas.
    
    Args:
        file_content (BytesIO): Il contenuto del file Excel in memoria.

    Returns:
        pd.DataFrame: Un DataFrame di Pandas con le colonne 'input_text' e 'target_text'.
    """
    try:
        corpus_list = []
        # Usa openpyxl per ottenere i nomi dei fogli
        workbook = openpyxl.load_workbook(file_content, read_only=True)
        sheet_names = workbook.sheetnames
        file_content.seek(0) # Riavvolgi il buffer dopo la lettura
        
        for sheet_name in sheet_names:
            # Salta i fogli che non ci interessano
            if sheet_name.lower().startswith(('copertina', 'copia')):
                continue

            # Trova la riga dell'intestazione in modo dinamico
            header_row_index = find_header_row(file_content, sheet_name)
            file_content.seek(0) # Riavvolgi il buffer
            
            if header_row_index is None:
                continue

            # Legge il foglio con l'intestazione corretta
            df_sheet = pd.read_excel(file_content, sheet_name=sheet_name, header=header_row_index)
            file_content.seek(0) # Riavvolgi il buffer

            giudizio_col = find_giudizio_column(df_sheet)
            if not giudizio_col:
                continue

            # Rimuove le righe vuote e le colonne che non sono utili
            df_sheet = df_sheet.dropna(how='all', subset=[col for col in df_sheet.columns if df_sheet[col].notna().any()])
            other_cols = [col for col in df_sheet.columns if col != giudizio_col and 'Unnamed' not in str(col)]

            # Prepara la lista di dizionari per la creazione del dataset
            data_for_dataset = []
            for index, row in df_sheet.iterrows():
                prompt_parts = []
                for col in other_cols:
                    value = row.get(col)
                    if pd.notna(value) and str(value).strip():
                        prompt_parts.append(f"{col}: {str(value).strip()}")
                
                prompt_text = " ".join(prompt_parts)
                target_text = str(row[giudizio_col]).strip() if pd.notna(row[giudizio_col]) else ""

                if prompt_text and target_text:
                    data_for_dataset.append({
                        'input_text': prompt_text,
                        'target_text': target_text
                    })
            
            if data_for_dataset:
                corpus_list.extend(data_for_dataset)
        
        if not corpus_list:
            return pd.DataFrame()
            
        return pd.DataFrame(corpus_list)
        
    except Exception as e:
        st.error(f"Errore nella lettura del file: {e}")
        st.error(f"Traceback:\n{traceback.format_exc()}")
        return pd.DataFrame()

# ==============================================================================
# SEZIONE 4: INTERFACCIA UTENTE (UI) E LOGICA PRINCIPALE
# ==============================================================================

st.title("Costruttore di Corpus AI")
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
    # Aggiorna lo stato della sessione con i nuovi file
    for file in uploaded_files:
        st.session_state.uploaded_files_data[file.name] = file
    
    # Processa tutti i file memorizzati
    corpus_list = []
    
    with st.status("Elaborazione dei file...", expanded=True) as status:
        try:
            for file_name, file_data in st.session_state.uploaded_files_data.items():
                status.write(f"Elaborazione del file '{file_name}'...")
                
                # Legge il contenuto del file in un buffer di memoria
                file_buffer = BytesIO(file_data.getvalue())
                
                df_new = load_and_prepare_excel(file_buffer)
                
                if not df_new.empty:
                    corpus_list.append(df_new)
                    status.write(f"File '{file_name}' elaborato con successo. Righe aggiunte: {len(df_new)}")
                else:
                    status.write(f"ATTENZIONE: Il file '{file_name}' non contiene dati validi. Saltato.")

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
    
