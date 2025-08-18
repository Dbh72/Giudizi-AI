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
import time

# Ignoriamo i FutureWarning per mantenere la console pulita.
warnings.filterwarnings("ignore")

# ==============================================================================
# SEZIONE 2: FUNZIONI AUSILIARIE PER LA GENERAZIONE
# ==============================================================================

def _process_text_in_chunks(model, tokenizer, input_text, max_length=512, chunk_overlap=50):
    """
    Processa un testo di input troppo lungo suddividendolo in chunk e generando
    una risposta per ogni chunk, poi riassembla le risposte.

    Args:
        model (PeftModel): Il modello fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        input_text (str): Il testo di input da elaborare.
        max_length (int): La lunghezza massima di input per il modello.
        chunk_overlap (int): La sovrapposizione tra i chunk.

    Returns:
        str: Il testo del giudizio generato.
    """
    input_tokens = tokenizer(
        input_text,
        max_length=max_length,
        truncation=True,
        return_overflowing_tokens=True,
        stride=chunk_overlap,
        padding="max_length"
    )
    
    generated_chunks = []
    
    # Processa ogni chunk e concatena i risultati
    for chunk_id in input_tokens['input_ids']:
        chunk_text = tokenizer.decode(chunk_id, skip_special_tokens=True)
        # Genera un giudizio per il singolo chunk
        input_ids = tokenizer(
            chunk_text,
            return_tensors="pt"
        ).input_ids.to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_chunks.append(generated_text)
        
    return " ".join(generated_chunks)

def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, progress_container, output_dir):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro selezionato.
        progress_container (gr.Textbox): Componente Gradio per i messaggi di stato.
        output_dir (str): Directory del modello fine-tuned.

    Returns:
        pd.DataFrame: Il DataFrame completato con i giudizi.
    """
    state_file = f"state_gen_{selected_sheet}.json"
    
    # Inizializza o carica lo stato del processo
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            last_completed_row = state.get("last_completed_row", -1)
            progress_container.append(f"Trovato uno stato precedente. Riprendo dalla riga {last_completed_row + 2}...")
            # Aggiungiamo una colonna di stato per tracciare le righe elaborate
            df_to_complete['generation_status'] = df_to_complete.index.map(lambda i: 'completed' if i <= last_completed_row else 'pending')
    else:
        last_completed_row = -1
        state = {"last_completed_row": -1}
        df_to_complete['generation_status'] = 'pending'
    
    # Iteriamo sul DataFrame, saltando le righe già completate
    for index, row in df_to_complete.iterrows():
        # Saltiamo le righe con un giudizio già presente o già completate in una run precedente
        if pd.notna(row[giudizio_col]) or df_to_complete.loc[index, 'generation_status'] == 'completed':
            progress_container.append(f"Riga {index + 2}: Giudizio già presente. Saltato.")
            continue
        
        progress_container.append(f"Riga {index + 2}: Generazione del giudizio in corso...")
        
        try:
            # Crea il prompt combinando le informazioni della riga
            prompt_text = "Giudizio: " + " ".join([str(v) for k, v in row.items() if k != giudizio_col and pd.notna(v)])
            
            # Controlla la lunghezza del prompt e applica il chunking se necessario
            max_model_length = 512 # Esempio di lunghezza massima, può variare
            if len(tokenizer.encode(prompt_text, max_length=10000)) > max_model_length:
                progress_container.append(f"Testo per la riga {index + 2} è troppo lungo. Applico la logica di chunking...")
                generated_text = _process_text_in_chunks(model, tokenizer, prompt_text, max_length=max_model_length)
            else:
                # Caso di generazione standard
                input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_length=150,
                        num_beams=4,
                        early_stopping=True
                    )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Aggiorna il DataFrame con il giudizio generato
            df_to_complete.loc[index, giudizio_col] = generated_text
            
            # Aggiorna il file di stato con l'ultimo indice completato
            state["last_completed_row"] = index
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            progress_container.append(f"Giudizio per la riga {index + 2} generato e salvato.")

        except Exception as e:
            error_message = f"Errore durante la generazione per la riga {index + 2}: {e}\n\nTraceback:\n{traceback.format_exc()}"
            progress_container.append(error_message)
            print(error_message)
            # Continua con la riga successiva
            continue
    
    # Rimuove la colonna di stato e il file di stato una volta completata la generazione
    print("Generazione completata con successo!")
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
        
        print(f"Modello e tokenizer caricati con successo da: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"Errore nel caricare il modello o il tokenizer: {e}")
        print(traceback.format_exc())
        return None, None
