from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

BOOKS = {
    84: "Frankenstein",
    345: "Dracula",
    43: "Dr Jekyll and Mr Hyde",
    174: "The Picture of Dorian Gray",
    36: "The War of the Worlds",
    35: "The Time Machine"
}

