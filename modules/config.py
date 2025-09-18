# Configuration constants
import os
import streamlit as st

LANGUAGE_CODE_MAPPING = {
    'en': 'eng_Latn',
    'ko': 'kor_Hang',
    'fr': 'fra_Latn',
    'es': 'spa_Latn',
    'it': 'ita_Latn',
    'pt': 'por_Latn',
    'ru': 'rus_Cyrl',
    'ja': 'jpn_Jpan',
    'hi': 'hin_Deva',
}

MODEL_CONFIGS = {
    'standard': {
        'name': 'facebook/nllb-200-distilled-600M',
        'label': 'Standard (600M) - Balanced speed and accuracy'
    },
    'm2m418': {
        'name': 'facebook/m2m100_418M',
        'label': 'M2M100 (418M) - Smaller, multilingual baseline'
    }
}

def get_default_model():
    """
    Decide which model to use by default.
    - On Streamlit Cloud: force M2M100 (418M)
    - Local: allow Standard (600M)
    """
    if "STREAMLIT_RUNTIME" in os.environ or "secrets" in dir(st):  # running on Streamlit Cloud
        return "m2m418"
    return "standard"

LOG_FILE = "jana_app.log"
CACHE_DB = "translation_cache.sqlite"
