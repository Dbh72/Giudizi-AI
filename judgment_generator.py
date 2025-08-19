# ==============================================================================
# File: judgment_generator.py
# Modulo per la generazione dei giudizi utilizzando un modello fine-tuned.
# ==============================================================================

# SEZIONE 1: LIBRERIE NECESSARIE
# ==============================================================================
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
from config import OUTPUT_DIR, MODEL_NAME

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
        max_length (int): La lunghezza massima di input per ogni chunk.
        chunk_overlap (int): Il numero di token sovrapposti tra i chunk.

    Returns:
        str: Il giudizio generato.
    """
    # Tokenizza il testo di input
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=False).input_ids
    
    # Calcola il numero di chunk necessari
    num_tokens = input_ids.shape[1]
    if num_tokens <= max_length:
        # Se il testo è abbastanza corto, processalo in un'unica soluzione
        generation_output = model.generate(
            input_ids=input_ids.to(model.device),
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        return tokenizer.decode(generation_output[0], skip_special_tokens=True)
        
    # Processa in chunk se il testo è troppo lungo
    all_generated_tokens = []
    current_idx = 0
    while current_idx < num_tokens:
        end_idx = min(current_idx + max_length, num_tokens)
        chunk_input_ids = input_ids[:, current_idx:end_idx].to(model.device)
        
        generation_output = model.generate(
            input_ids=chunk_input_ids,
            max_new_tokens=1024,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        all_generated_tokens.append(generated_text)
        
        # Sposta l'indice per il prossimo chunk con sovrapposizione
        current_idx += (max_length - chunk_overlap)
        
    # Assembla le risposte dei chunk (logica semplificata per dimostrazione)
    # Una logica più complessa potrebbe usare tecniche di riassunto o coerenza
    return " ".join(all_generated_tokens)

def generate_judgments_for_excel(model, tokenizer, df_to_complete, giudizio_col, selected_sheet, output_dir, progress_container, batch_size=16):
    """
    Genera i giudizi per un DataFrame, aggiungendo la logica di resumibilità e checkpoint.

    Args:
        model (PeftModel): Il modello PEFT fine-tuned.
        tokenizer (AutoTokenizer): Il tokenizer del modello.
        df_to_complete (pd.DataFrame): Il DataFrame da completare.
        giudizio_col (str): Il nome della colonna 'Giudizio'.
        selected_sheet (str): Il nome del foglio di lavoro.
        output_dir (str): Il percorso della directory di output.
        progress_container (callable): Funzione per inviare messaggi di stato.
        batch_size (int): Numero di righe da processare in un singolo batch.
    
    Returns:
        pd.DataFrame: Il DataFrame completato.
    """
    state_file = os.path.join(output_dir, f"state_{selected_sheet}.json")

    # Aggiungi una colonna temporanea per lo stato di generazione
    df_to_complete['generation_status'] = 'pending'

    # Carica lo stato precedente se esiste
    start_index = 0
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            if state.get('sheet') == selected_sheet:
                start_index = state.get('last_completed_index', 0) + 1
                if start_index < len(df_to_complete):
                    progress_container(f"Trovato uno stato precedente. Riprendo la generazione dal punto in cui era stata interrotta (riga {start_index+2}).", "info")
                    # Segna le righe precedenti come completate per non rigenerarle
                    df_to_complete.loc[:start_index, 'generation_status'] = 'completed'
                else:
                    progress_container("Il processo precedente sembra già completato. Nessuna riga da generare.", "warning")
                    return df_to_complete.drop(columns=['generation_status'])

    # Ottimizzazione: processa in batch
    total_rows = len(df_to_complete)
    progress_container(f"Inizio della generazione di {total_rows - start_index} giudizi (batch size: {batch_size})...", "info")
    
    for i in range(start_index, total_rows, batch_size):
        batch_df = df_to_complete.iloc[i:i + batch_size]
        
        # Prepara il prompt per ogni riga nel batch
        prompts = []
        original_indices = []
        for index, row in batch_df.iterrows():
            if pd.notna(row['generation_status']) and row['generation_status'] == 'completed':
                continue # Salta le righe già completate
            
            # Crea il prompt combinando tutte le colonne tranne 'Giudizio'
            input_data = row.drop(labels=[giudizio_col, 'generation_status'])
            prompt_text = " ".join([f"{col}: {str(val)}" for col, val in input_data.items() if pd.notna(val)])
            prompts.append(prompt_text)
            original_indices.append(index)

        if not prompts:
            continue

        progress_container(f"Processando batch {i // batch_size + 1} di {total_rows // batch_size + 1}...", "info")
        
        try:
            # Codifica i prompt in batch
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
            inputs.to(model.device)
            
            # Generazione del giudizio in batch
            generated_tokens = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )
            
            # Decodifica e assegna i giudizi
            generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for j, text in enumerate(generated_texts):
                original_index = original_indices[j]
                df_to_complete.at[original_index, giudizio_col] = text
                df_to_complete.at[original_index, 'generation_status'] = 'completed'

            # Salva lo stato dopo ogni batch
            state = {
                'sheet': selected_sheet,
                'last_completed_index': max(original_indices),
                'timestamp': datetime.now().isoformat()
            }
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            time.sleep(1) # Pausa per visualizzazione del progresso
            
        except Exception as e:
            error_message = f"Errore durante la generazione per il batch a partire dalla riga {i + 2}: {e}"
            progress_container(error_message, "error")
            progress_container(f"Traceback: {traceback.format_exc()}", "error")
            # Rimuoviamo la colonna di stato prima di restituire il DF
            return df_to_complete.drop(columns=['generation_status'])
            
    # Rimuove la colonna di stato una volta completata la generazione
    progress_container("Generazione completata con successo!", "success")
    # Rimuovi il file di stato una volta completato il processo
    if os.path.exists(state_file):
        os.remove(state_file)
    return df_to_complete.drop(columns=['generation_status'])

def load_trained_model(model_path, progress_container):
    """
    Carica il modello e il tokenizer fine-tuned, controllando se il percorso
    esiste localmente.

    Args:
        model_path (str): Il percorso della directory del modello salvato.
        progress_container (callable): Funzione per inviare messaggi di stato.

    Returns:
        tuple: (model, tokenizer) o (None, None) se il caricamento fallisce.
    """
    try:
        progress_container(f"Caricamento del modello da: {model_path}...", "info")
        
        # Controlla se il modello esiste localmente
        if not os.path.exists(model_path):
            progress_container(f"Errore: La directory del modello '{model_path}' non è stata trovata localmente.", "error")
            return None, None
            
        # Carica il tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Carica il modello base e applica gli adattatori PEFT
        base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto")
        model = PeftModel.from_pretrained(base_model, model_path)
        
        progress_container(f"Modello e tokenizer caricati con successo da: {model_path}", "success")
        return model, tokenizer

    except Exception as e:
        progress_container(f"Errore durante il caricamento del modello: {e}", "error")
        progress_container(f"Traceback: {traceback.format_exc()}", "error")
        return None, None
