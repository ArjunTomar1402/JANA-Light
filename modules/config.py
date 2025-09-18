# Configuration constants
import os
import streamlit as st

LANGUAGE_CODE_MAPPING = {
    "en": "en",
    "hi": "hi",
    "ja": "ja",
    "ko": "ko",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "pt": "pt",
    "ru": "ru"
}

MODEL_CONFIGS = {
    'm2m418': {
        'name': 'facebook/m2m100_418M',
        'label': 'M2M100 (418M) - Smaller, multilingual baseline'
    }
}

LOG_FILE = "jana_app.log"
CACHE_DB = "translation_cache.sqlite"
