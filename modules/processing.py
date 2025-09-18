import streamlit as st
import re
import torch  
from typing import List  
from sudachipy import SplitMode
from modules.models import get_models
from modules.translation import translate_text
from modules.utils import generate_furigana, _sudachi_to_string, is_japanese
from modules.config import MODEL_CONFIGS, LANGUAGE_CODE_MAPPING

# Mapping of Unicode ranges for language fallback
LANGUAGE_UNICODE_RANGES = {
    'en': r'[A-Za-z]',
    'hi': r'[\u0900-\u097F]',
    'ja': r'[\u3040-\u30FF\u4E00-\u9FFF]',
    'ko': r'[\uAC00-\uD7AF]',
    'fr': r'[A-Za-zÀ-ÿ]',
    'es': r'[A-Za-zÀ-ÿ]',
    'it': r'[A-Za-zÀ-ÿ]',
    'pt': r'[A-Za-zÀ-ÿ]',
    'ru': r'[\u0400-\u04FF]',
}

CONFIDENCE_THRESHOLD = 0.5

def process_sentence(sentence: str) -> dict:
    """Process a single sentence"""
    lid_model, translator_model, translator_tokenizer, sudachi_tokenizer_obj, _ = get_models()
    
    try:
        clean_sentence = sentence.replace("\n", " ").strip()
        if not clean_sentence:
            return None

        # Language detection
        predictions = lid_model.predict(clean_sentence, k=1)
        lang_code = predictions[0][0].replace('__label__', '')
        conf = float(predictions[1][0])

        # Confidence threshold fallback
        if conf < CONFIDENCE_THRESHOLD or lang_code not in LANGUAGE_CODE_MAPPING:
            for lc, regex in LANGUAGE_UNICODE_RANGES.items():
                if re.search(regex, clean_sentence):
                    lang_code = lc
                    conf = 0.6  # default fallback confidence
                    break
            else:
                lang_code = 'en'
                conf = 0.6

        if st.session_state.get('debug_mode', False):
            st.write(f"[DEBUG] Sentence: {clean_sentence}  Detected: {lang_code} ({conf:.2f})")

        if lang_code == 'ja':
            morphemes = sudachi_tokenizer_obj.tokenize(clean_sentence, SplitMode.C)
            tokenized_output = _sudachi_to_string(morphemes)
            furigana = generate_furigana(clean_sentence) if st.session_state.get('generate_furigana', False) else ""
            return {
                "Original Text": sentence,
                "Detected Language": "Japanese",
                "Confidence": f"{conf:.2f}",
                "Standard Japanese": clean_sentence,
                "Furigana": furigana,
                "Morphological Analysis": tokenized_output
            }
        else:
            model_name_for_cache = st.session_state.get('translator_name', MODEL_CONFIGS['standard']['name'])
            jp_translation = translate_text(clean_sentence, lang_code, model_name_for_cache)

            # Error handling
            if jp_translation.startswith("[Translation error:") or jp_translation.startswith("[Rate limit"):
                return {
                    "Original Text": sentence,
                    "Detected Language": lang_code,
                    "Confidence": f"{conf:.2f}",
                    "Standard Japanese": jp_translation,
                    "Furigana": "",
                    "Morphological Analysis": ""
                }

            morphemes = sudachi_tokenizer_obj.tokenize(jp_translation, SplitMode.C)
            tokenized_output = _sudachi_to_string(morphemes)
            furigana = generate_furigana(jp_translation) if st.session_state.get('generate_furigana', False) else ""

            if not is_japanese(jp_translation):
                jp_translation = "[NOT JAPANESE OUTPUT] " + jp_translation

            return {
                "Original Text": sentence,
                "Detected Language": lang_code,
                "Confidence": f"{conf:.2f}",
                "Standard Japanese": jp_translation,
                "Furigana": furigana,
                "Morphological Analysis": tokenized_output
            }

    except Exception as e:
        return {
            "Original Text": sentence,
            "Detected Language": "Error",
            "Confidence": "0.00",
            "Standard Japanese": f"Error: {e}",
            "Furigana": "",
            "Morphological Analysis": ""
        }

def process_text_batch(sentences: List[str], batch_size: int = 1) -> List[dict]:
    """Process text in batches"""
    results = []
    total_sentences = len(sentences)
    if total_sentences == 0:
        return results

    progress_bar = st.progress(0, text="Processing...")
    status_text = st.empty()

    for i, sentence in enumerate(sentences):
        progress = (i + 1) / total_sentences
        progress_bar.progress(progress, text=f"Processed {i+1}/{total_sentences} sentences")
        status_text.text(f"Processing sentence {i+1}/{total_sentences}")

        result = process_sentence(sentence)
        if result:
            results.append(result)

        # Release GPU memory if needed
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    progress_bar.empty()
    status_text.empty()
    return results
