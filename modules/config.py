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
    'm2m418': {
        'name': 'facebook/m2m100_418M',
        'label': 'M2M100 (418M) - Smaller, multilingual baseline'
    }
}

LOG_FILE = "jana_app.log"
CACHE_DB = "translation_cache.sqlite"
