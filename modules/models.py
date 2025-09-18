import streamlit as st
import fasttext
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from sudachipy import dictionary
import pykakasi
import torch
from modules.config import MODEL_CONFIGS
from modules.utils import download_fasttext_model  # ✅ now correct

# Global model references
lid_model = None
translator_model = None
translator_tokenizer = None
sudachi_tokenizer_obj = None
kakasi_instance = None


@st.cache_resource
def load_models(model_size='standard', device_name: str = "cpu", custom_model_name: str = None):
    """Load all required models"""
    progress_bar = st.progress(0, text="Loading models...")
    models = {}

    # Load FastText
    progress_bar.progress(10, text="Loading language detection model...")
    try:
        model_path = download_fasttext_model()
        if model_path:
            models['lid_model'] = fasttext.load_model(model_path)
        else:
            st.error("Could not load fasttext model")
            models['lid_model'] = None
    except Exception as e:
        st.error(f"Could not load fasttext model: {e}")
        models['lid_model'] = None

    # Load translation model
    progress_bar.progress(40, text="Loading translation model...")
    try:
        model_name = custom_model_name or MODEL_CONFIGS.get(model_size, MODEL_CONFIGS['standard'])['name']

        if "m2m100" in model_name.lower():
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        models['translator_tokenizer'] = tokenizer
        models['translator_model'] = model.to(device_name)
        models['translator_name'] = model_name
    except Exception as e:
        st.error(f"Could not load translation model '{model_name}': {e}")
        models['translator_tokenizer'] = None
        models['translator_model'] = None
        models['translator_name'] = None

    # Load Sudachi
    progress_bar.progress(70, text="Loading morphological analyzer...")
    try:
        models['sudachi_tokenizer_obj'] = dictionary.Dictionary().create()
    except Exception as e:
        st.error(f"Could not initialize Sudachi: {e}")
        models['sudachi_tokenizer_obj'] = None

    # Load PyKakasi
    progress_bar.progress(90, text="Loading furigana generator...")
    try:
        models['kakasi'] = pykakasi.kakasi()
    except Exception as e:
        st.warning(f"Could not initialize kakasi for furigana: {e}")
        models['kakasi'] = None

    progress_bar.progress(100, text="Models loaded (with warnings if any).")
    return models


def get_models():
    """Return global model references"""
    return lid_model, translator_model, translator_tokenizer, sudachi_tokenizer_obj, kakasi_instance


def set_models(lid, translator, tokenizer, sudachi, kakasi):
    """Set global model references"""
    global lid_model, translator_model, translator_tokenizer, sudachi_tokenizer_obj, kakasi_instance
    lid_model = lid
    translator_model = translator
    translator_tokenizer = tokenizer
    sudachi_tokenizer_obj = sudachi
    kakasi_instance = kakasi
