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
import json

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
        selected_sheet (str): Il nome del foglio di lavoro.
        output_dir (str): La directory dove salvare lo stato del processo.

    Returns:
        pd.DataFrame: Il DataFrame completato.
    """
    print("Avvio della generazione dei giudizi...")
    
    # Aggiungi una colonna temporanea per lo stato
    df_to_complete['generation_status'] = 'Pending'
    
    state_file = os.path.join(output_dir, f'progress_state_{selected_sheet}.json')
    last_processed_index = -1
    
    # Carica lo stato precedente se esiste
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_processed_index = state.get('last_processed_index', -1)
            
            print(f"Trovato stato precedente. Riprendo dalla riga {last_processed_index + 1}.")
            
            # Carica il DataFrame salvato se esiste
            last_df_path = state.get('last_df_path')
            if last_df_path and os.path.exists(last_df_path):
                df_to_complete = pd.read_csv(last_df_path)
            
        except Exception as e:
            print(f"Errore nel caricamento dello stato, riavvio da zero: {e}")
            os.remove(state_file)
            
    # Iteriamo sulle righe per generare i giudizi
    for index, row in df_to_complete.iterrows():
        # Saltiamo le righe già processate
        if index <= last_processed_index:
            print(f"Salto la riga {index}, già processata.")
            continue
            
        # Saltiamo la riga se la colonna Giudizio è già compilata
        if pd.notna(row[giudizio_col]) and str(row[giudizio_col]).strip() != "":
            print(f"Salto la riga {index}, il giudizio è già presente.")
            df_to_complete.at[index, 'generation_status'] = 'Skipped (already exists)'
            continue
            
        try:
            prompt_parts = []
            for col, value in row.items():
                if col != giudizio_col and pd.notna(value) and str(value).strip():
                    prompt_parts.append(f"{col}: {str(value).strip()}")
            
            prompt_text = " ".join(prompt_parts)
            
            if not prompt_text:
                print(f"Attenzione: Riga {index} vuota o non valida, salto la generazione.")
                df_to_complete.at[index, 'generation_status'] = 'Skipped (empty row)'
                continue

            # Tokenizzazione e generazione del testo
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Assegnazione del giudizio generato
            df_to_complete.at[index, giudizio_col] = generated_text
            df_to_complete.at[index, 'generation_status'] = 'Completed'
            print(f"Generato giudizio per la riga {index}.")
            
            # Salvataggio dello stato e del DataFrame ogni 50 righe (checkpoint)
            if index % 50 == 0:
                print(f"Checkpoint di generazione: salvataggio dello stato al passo {index}...")
                temp_df_path = os.path.join(output_dir, f'temp_df_{selected_sheet}.csv')
                df_to_complete.to_csv(temp_df_path, index=False)
                state = {'last_processed_index': index, 'last_df_path': temp_df_path}
                with open(state_file, 'w') as f:
                    json.dump(state, f)

        except Exception as e:
            print(f"Errore durante la generazione per la riga {index}: {e}")
            df_to_complete.at[index, 'generation_status'] = 'Error'
            # Salvataggio dello stato per riprendere da qui
            temp_df_path = os.path.join(output_dir, f'temp_df_{selected_sheet}.csv')
            df_to_complete.to_csv(temp_df_path, index=False)
            state = {'last_processed_index': index, 'last_df_path': temp_df_path}
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            # Rimuoviamo la colonna di stato prima di restituire il DF
            return df_to_complete.drop(columns=['generation_status'])
            
    # Rimuove la colonna di stato una volta completata la generazione
    print("Generazione completata con successo!")
    # Rimuovi il file di stato una volta completato il processo
    if os.path.exists(state_file):
        os.remove(state_file)
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
        print(f"Caricamento del modello da: {model_path}...")
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Carica il modello base e applica gli adattatori PEFT
        base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print(f"Modello e tokenizer caricati con successo.")
        return model, tokenizer
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        print(traceback.format_exc())
        return None, None
