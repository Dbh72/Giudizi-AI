# app.py - Orchestratore principale
# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import os
import pandas as pd
import gradio as gr
import shutil
import json
from datetime import datetime
import traceback
import time
import pickle

# Importiamo i moduli "reader" dedicati.
# Per ora abbiamo solo excel_reader, ma potremmo aggiungere altri in futuro.
import excel_reader as reader_excel

# ==============================================================================
# SEZIONE 2: CONFIGURAZIONE GLOBALE E CARICAMENTO MODELLO
# ==============================================================================
# Directory dove il modello addestrato verrà salvato.
OUTPUT_DIR = "modello_finetunato"
# Il modello base che stiamo usando per il fine-tuning.
MODEL_NAME = "google/flan-t5-small"

# Le variabili globali per il modello e il tokenizer.
# Vengono inizializzate una volta sola all'avvio dell'app.
# La logica completa è stata spostata in una funzione dedicata.
fine_tuned_model = None
tokenizer = None

def load_initial_model():
    """Carica il modello base e il tokenizer all'avvio dell'app."""
    global tokenizer, fine_tuned_model
    try:
        # Controlliamo se è già stato fatto il fine-tuning e carichiamo il modello appropriato
        if os.path.exists(OUTPUT_DIR):
            gr.Warning("Modello finetunato esistente, lo sto caricando...")
            from peft import PeftModel, LoraConfig, get_peft_model, TaskType
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, load_in_8bit=True, device_map="auto")
            fine_tuned_model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
            fine_tuned_model = fine_tuned_model.merge_and_unload()
            gr.Info("Modello finetunato caricato con successo!")
            return "ready"
        else:
            gr.Info("Modello base non ancora finetunato. Carica un file per iniziare.")
            return "not_ready"
    except Exception as e:
        gr.Error(f"Errore nel caricamento del modello iniziale: {e}")
        return "error"

# ==============================================================================
# SEZIONE 3: LOGICA DI ORCHESTRAZIONE DEL CORPO
# ==============================================================================

def orchestrate_data_pipeline(uploaded_files):
    """
    Funzione principale per gestire il caricamento e la preparazione di tutti i file.
    Agisce come l'orchestratore descritto nel piano di architettura.
    
    Args:
        uploaded_files (list): Una lista di oggetti Gradio File.
        
    Returns:
        tuple: (stringa di stato, DataFrame unificato)
    """
    if not uploaded_files:
        return "Nessun file caricato.", None
        
    all_dfs = []
    status_messages = []
    
    for file in uploaded_files:
        try:
            file_path = file.name
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Identifichiamo il tipo di file e chiamiamo il reader dedicato
            if file_extension in ['.xlsx', '.xls', '.xlsm']:
                gr.Info(f"Elaborazione del file Excel: {os.path.basename(file_path)}")
                dfs_from_excel = reader_excel.load_and_prepare_excel(file_path)
                
                if dfs_from_excel:
                    for sheet_name, df in dfs_from_excel.items():
                        all_dfs.append(df)
                        status_messages.append(f"Foglio '{sheet_name}' da '{os.path.basename(file_path)}' caricato.")
                else:
                    status_messages.append(f"Errore nell'elaborazione del file Excel: {os.path.basename(file_path)}")
            
            # Qui si potrebbero aggiungere altri reader in futuro
            elif file_extension == '.docx':
                status_messages.append(f"Gestione file .docx non ancora implementata per {os.path.basename(file_path)}")
            elif file_extension == '.txt':
                status_messages.append(f"Gestione file .txt non ancora implementata per {os.path.basename(file_path)}")
            else:
                status_messages.append(f"Tipo di file non supportato: {os.path.basename(file_path)}")
        
        except Exception as e:
            status_messages.append(f"Errore critico durante l'elaborazione di {os.path.basename(file_path)}: {e}")
            traceback.print_exc()
    
    final_corpus = pd.DataFrame()
    if all_dfs:
        try:
            # Concateniamo tutti i DataFrame in un unico corpus
            final_corpus = pd.concat(all_dfs, ignore_index=True)
            status_messages.append("Corpus di dati unificato con successo!")
        except Exception as e:
            status_messages.append(f"Errore nella concatenazione dei DataFrame: {e}")
            
    return "\n".join(status_messages), final_corpus


# ==============================================================================
# SEZIONE 4: FUNZIONI DELL'INTERFACCIA GRADIO
# ==============================================================================

# Definisco una variabile di stato per il processo di fine-tuning
fine_tuning_state = "not_started"

def fine_tune_model(file_paths, fine_tuning_state):
    """
    Funzione per l'addestramento del modello.
    Ora riceve il percorso di un file.
    """
    if fine_tuning_state == "running":
        gr.Warning("Il fine-tuning è già in corso. Attendi che finisca.")
        return "Fine-tuning già in esecuzione.", fine_tuning_state

    if not file_paths:
        gr.Warning("Per favore, carica almeno un file per avviare il fine-tuning.")
        return "Nessun file caricato per il fine-tuning.", "not_started"
        
    try:
        gr.Info("Avvio della pipeline di orchestrazione...")
        status, corpus = orchestrate_data_pipeline(file_paths)
        gr.Info(status)
        
        if corpus is None or corpus.empty:
            gr.Error("Impossibile creare il corpus di dati per il fine-tuning.")
            return "Fine-tuning fallito: corpus vuoto.", "failed"

        # Simula il processo di fine-tuning con il corpus unificato
        gr.Info(f"Avvio del fine-tuning con un corpus di {len(corpus)} record.")
        time.sleep(5)  # Simula l'addestramento
        gr.Info("Fine-tuning completato. Modello salvato.")
        
        # Simula il salvataggio del modello
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, "fake_model.pkl"), "wb") as f:
            pickle.dump(corpus, f) # Salviamo il corpus per dimostrazione

        global fine_tuned_model
        fine_tuned_model = "fake_model_ready" # Segnaliamo che il modello è pronto

        return "Fine-tuning completato con successo!", "completed"

    except Exception as e:
        traceback.print_exc()
        gr.Error(f"Si è verificato un errore durante il fine-tuning: {e}")
        return "Fine-tuning fallito.", "failed"

def generate_judgment(prompt, model_state):
    """Genera un giudizio basato sul prompt e sullo stato del modello."""
    if model_state == "not_ready":
        gr.Warning("Il modello non è ancora pronto. Per favore, addestralo prima.")
        return "Il modello non è pronto."

    # Qui usiamo un placeholder per la generazione
    return f"Giudizio generato per: '{prompt}'"


# ==============================================================================
# SEZIONE 5: INTERFACCIA GRADIO
# ==============================================================================
with gr.Blocks(title="Generatore Giudizi AI") as demo:
    gr.Markdown("# Generatore di Giudizi di Matematica con IA")
    gr.Markdown("Carica i tuoi file Excel per addestrare l'IA. Dopo l'addestramento, potrai generare nuovi giudizi.")

    fine_tuning_state = gr.State(value=load_initial_model())

    with gr.Tab("Fine-Tuning e Addestramento"):
        gr.Markdown("### 🚀 Addestramento del Modello 🚀")
        gr.Markdown("Carica uno o più file Excel. Il modello utilizzerà i dati di tutti i file per l'addestramento.")

        with gr.Row():
            fine_tune_file_input = gr.File(label="Carica i tuoi file di addestramento", file_count="multiple", file_types=[".xlsx", ".xls", ".xlsm", ".docx", ".txt"])
            fine_tune_button = gr.Button("Avvia Fine-Tuning")
        
        fine_tune_status_output = gr.Textbox(label="Stato del Fine-Tuning", interactive=False, lines=3)
        
    with gr.Tab("Genera Giudizi"):
        gr.Markdown("### ✍️ Genera un nuovo giudizio ✍️")
        gr.Markdown("Inserisci una descrizione della verifica (es: 'ha svolto bene le equazioni, ma ha difficoltà con i problemi con le frazioni').")
        
        with gr.Row():
            prompt_input = gr.Textbox(label="Descrizione della verifica", lines=2)
            generate_button = gr.Button("Genera Giudizio")
        
        giudizio_output = gr.Textbox(label="Giudizio Generato", lines=5, interactive=False)

    fine_tune_button.click(
        fn=fine_tune_model,
        inputs=[fine_tune_file_input, fine_tuning_state],
        outputs=[fine_tune_status_output, fine_tuning_state]
    )
    
    generate_button.click(
        fn=generate_judgment,
        inputs=[prompt_input, fine_tuning_state],
        outputs=giudizio_output
    )

    # Inizializzazione e gestione dello stato
    demo.load(
        fn=lambda: gr.Info("Applicazione pronta. Carica i file per iniziare il fine-tuning."),
        inputs=None,
        outputs=None
    )


# excel_reader.py - Modulo per la lettura dei file Excel
# ==============================================================================
# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
import pandas as pd
import openpyxl
import os

# ==============================================================================
# SEZIONE 2: LOGICA DI GESTIONE DEI DATI
# ==============================================================================

def detect_header_row(df_sheet):
    """
    Individua la riga di intestazione (header) basandosi sull'assenza di valori numerici.
    
    Questa funzione analizza le prime 10 righe di un DataFrame e restituisce l'indice
    della prima riga che non contiene solo valori numerici.
    Questo aiuta a gestire i file Excel che hanno metadati o righe vuote prima dell'intestazione.
    """
    for i in range(min(10, len(df_sheet))):
        if not df_sheet.iloc[i].apply(lambda x: isinstance(x, (int, float))).all():
            return i
    return 0

def load_and_prepare_excel(file_path):
    """
    Carica un file Excel e prepara i dati per l'addestramento.
    
    Questa funzione legge tutti i fogli di lavoro da un file Excel, identifica
    l'intestazione e restituisce un dizionario con i DataFrame per ogni foglio.
    Viene utilizzata la stessa logica di lettura presente nel file 33 Funziona.txt.
    
    Args:
        file_path (str): Il percorso del file Excel.
        
    Returns:
        dict: Un dizionario dove le chiavi sono i nomi dei fogli e i valori sono i 
              rispettivi DataFrame. Ritorna None in caso di errore.
    """
    try:
        # Usiamo pd.ExcelFile per leggere il file e ottenere i nomi di tutti i fogli
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        extracted_dfs = {}
        for sheet_name in sheet_names:
            # Per ogni foglio, lo leggiamo come un DataFrame
            df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            
            # Troviamo la riga di intestazione
            header_row_index = detect_header_row(df_sheet)
            
            # Assegniamo l'intestazione e rimuoviamo le righe sopra di essa
            df_sheet.columns = df_sheet.iloc[header_row_index]
            df_sheet = df_sheet[header_row_index + 1:].reset_index(drop=True)
            
            extracted_dfs[sheet_name] = df_sheet
        
        return extracted_dfs
    
    except Exception as e:
        print(f"Errore nel caricamento del file Excel {file_path}: {e}")
        return None
