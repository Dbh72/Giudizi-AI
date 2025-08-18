# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando un modello fine-tuned.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
# Importiamo le librerie necessarie per la generazione e la gestione dei file.
import os
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import pandas as pd
import traceback
from datetime import datetime

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: FUNZIONI PER LA GENERAZIONE
# ==============================================================================

def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, output_dir):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro selezionato.
        output_dir (str): La directory dove salvare i file di stato.

    Returns:
        pd.DataFrame: Il DataFrame completato con i giudizi generati.
    """
    # Aggiunge una colonna di stato per tracciare i progressi
    df_to_complete['generation_status'] = ''
    
    # Crea una directory per i file di checkpoint se non esiste
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Carica lo stato precedente se esiste
    state_file = os.path.join(checkpoint_dir, f'generation_state_{selected_sheet}.json')
    last_processed_index = -1
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_processed_index = state.get('last_processed_index', -1)
                # Potremmo anche caricare l'intero df salvato
                if os.path.exists(state.get('last_df_path')):
                    df_to_complete = pd.read_csv(state.get('last_df_path'))
                    print(f"Ripresa da dove si era interrotto: dall'indice {last_processed_index + 1}.")
        except Exception as e:
            print(f"Errore nel caricamento dello stato. Riavvio la generazione. Errore: {e}")
            
    # Itera sulle righe per generare i giudizi
    for index, row in df_to_complete.iterrows():
        # Controlla se questa riga è già stata elaborata in un'esecuzione precedente
        if index <= last_processed_index:
            print(f"Saltata la riga {index + 1}, già elaborata.")
            continue
            
        print(f"Generazione giudizio per la riga {index + 1}...")
        
        try:
            # Crea il prompt di input unendo le altre colonne
            prompt_parts = []
            for col in df_to_complete.columns:
                if col != giudizio_col and col != 'generation_status':
                    value = row.get(col)
                    if pd.notna(value) and str(value).strip():
                        prompt_parts.append(f"{col}: {str(value).strip()}")
            
            prompt_text = " ".join(prompt_parts)
            
            # Genera il giudizio
            inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs['input_ids'])
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Aggiorna il DataFrame con il giudizio generato
            df_to_complete.loc[index, giudizio_col] = generated_text
            df_to_complete.loc[index, 'generation_status'] = 'Completato'
            
            # Checkpoint: salva lo stato ogni N righe (es. 50 righe)
            if (index + 1) % 50 == 0:
                print(f"Checkpoint! Salvataggio dello stato al passo {index + 1}...")
                temp_df_path = os.path.join(checkpoint_dir, f'temp_df_{selected_sheet}.csv')
                df_to_complete.to_csv(temp_df_path, index=False)
                
                state = {'last_processed_index': index, 'last_df_path': temp_df_path}
                with open(state_file, 'w') as f:
                    json.dump(state, f)
            
        except Exception as e:
            print(f"Errore durante la generazione per la riga {index + 1}. Salvataggio dello stato e interruzione.")
            traceback.print_exc()
            df_to_complete.loc[index, 'generation_status'] = 'Errore'
            
            # Salva lo stato e il file temporaneo anche in caso di errore
            temp_df_path = os.path.join(checkpoint_dir, f'temp_df_{selected_sheet}.csv')
            df_to_complete.to_csv(temp_df_path, index=False)
            state = {'last_processed_index': index, 'last_df_path': temp_df_path}
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            # Rimuoviamo la colonna di stato prima di restituire il DF
            return df_to_complete.drop(columns=['generation_status'])
            
    # Rimuove la colonna di stato una volta completata la generazione
    print("Generazione completata con successo!")
    return df_to_complete.drop(columns=['generation_status'])

def load_trained_model(model_path):
    """
    Carica il modello e il tokenizer fine-tuned.

    Args:
        model_path (str): Il percorso della directory del modello salvato.

    Returns:
        tuple: (model, tokenizer) o (None, None) se il caricamento fallisce.
    """
    try:
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Carica il modello base e applica gli adattatori PEFT
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"Modello e tokenizer caricati correttamente da {model_path}.")
        return model, tokenizer
    except Exception as e:
        print(f"Errore nel caricamento del modello da {model_path}.")
        traceback.print_exc()
        return None, None
