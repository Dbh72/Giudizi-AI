# ==============================================================================
# File: config.py
# File di configurazione per il progetto.
# ==============================================================================
# Importiamo le librerie necessarie
import os

# Definiamo il nome del modello di base da utilizzare
MODEL_NAME = "t5-small"

# Definiamo la directory di output per i file generati (modello, log, ecc.)
# L'uso di `os.path.join` garantisce la compatibilità tra i diversi sistemi operativi.
OUTPUT_DIR = "./modello_finetunato"

# Definiamo il nome del file del corpus, che conterrà i dati di addestramento.
CORPUS_FILE = "corpus.csv"

# Creiamo la directory di output se non esiste
os.makedirs(OUTPUT_DIR, exist_ok=True)
