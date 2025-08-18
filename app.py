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
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Importiamo il modulo con la logica per la preparazione dei dati
# Assicurati che 'excel_reader.py' si trovi nella stessa directory.
from excel_reader import load_and_prepare_excel, find_giudizio_column

# ==============================================================================
# SEZIONE 1: CONFIGURAZIONE GLOBALE E GESTIONE DELLO STATO
# ==============================================================================

OUTPUT_DIR = "modello_finetunato"
MODEL_NAME = "google/flan-t5-base"

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
# SEZIONE 2: FUNZIONI PER IL FINE-TUNING
# ==============================================================================
@st.cache_resource
def get_tokenizer_and_model():
    """Carica e restituisce il tokenizer e il modello base."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

def fine_tune_model(corpus_df):
    """
    Esegue il fine-tuning del modello sul corpus fornito.
    """
    try:
        if corpus_df.empty:
            st.error("Corpus vuoto. Carica file validi per procedere.")
            return

        tokenizer, model = get_tokenizer_and_model()

        # Prepara il dataset per il fine-tuning
        st.session_state.fine_tuning_state["current_step"] = "Preparazione del dataset..."
        st.session_state.fine_tuning_state["progress"] = 0.1
        
        dataset = Dataset.from_pandas(corpus_df)
        
        def tokenize_function(examples):
            return tokenizer(examples['input_text'], text_target=examples['target_text'], max_length=512, truncation=True)
            
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        # Configura PEFT per il fine-tuning con LoRa
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        model = get_peft_model(model, lora_config)

        # Configurazione degli argomenti di addestramento
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            report_to="none",
            save_strategy="epoch",  # Salva ad ogni epoca per il checkpoint
            load_best_model_at_end=False
        )

        # Inizializza il trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        st.session_state.fine_tuning_state["current_step"] = "Inizio addestramento..."
        st.session_state.fine_tuning_state["progress"] = 0.2
        
        # Avvia l'addestramento
        trainer.train()

        st.session_state.fine_tuning_state["current_step"] = "Addestramento completato. Salvataggio del modello..."
        st.session_state.fine_tuning_state["progress"] = 0.9

        # Salvataggio del modello
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        trainer.save_model(OUTPUT_DIR)
        
        st.session_state.fine_tuning_state["status"] = "Completato!"
        st.session_state.fine_tuning_state["progress"] = 1.0
        st.session_state.fine_tuning_state["current_step"] = "Modello pronto per la generazione."
        st.success("Fine-Tuning completato e modello salvato con successo!")

    except Exception as e:
        st.session_state.fine_tuning_state["status"] = "Errore"
        st.session_state.fine_tuning_state["current_step"] = f"Errore: {e}"
        st.error(f"Errore durante il fine-tuning: {e}\n\nTraceback:\n{traceback.format_exc()}")
        
# ==============================================================================
# SEZIONE 3: LAYOUT E INTERFACCIA UTENTE
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
                st.session_state.uploaded_files_data[file.name] = file.read()
            
            full_corpus_list = []
            for file_name, file_data in st.session_state.uploaded_files_data.items():
                # Chiamata alla funzione aggiornata con il nome del file
                df = load_and_prepare_excel(BytesIO(file_data), file_name)
                full_corpus_list.append(df)
            
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

    # Pulsante per avviare il fine-tuning
    if st.button("Avvia Fine-Tuning del Modello", disabled=st.session_state.corpus_df.empty):
        with st.spinner("Addestramento del modello in corso..."):
            fine_tune_model(st.session_state.corpus_df)

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
            df_to_complete = pd.read_excel(file_data, sheet_name=selected_sheet)
            giudizio_col = find_giudizio_column(df_to_complete)

            if giudizio_col is None:
                st.warning("La colonna 'Giudizio' non è stata trovata. Impossibile procedere.")
                return

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
