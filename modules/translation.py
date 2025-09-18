from transformers import pipeline
import torch
from modules.config import LANGUAGE_CODE_MAPPING
from modules.utils import cache_lookup, cache_store, get_rate_limiter, is_japanese, post_process_japanese
from modules.models import get_models
import re
import unicodedata

def normalize_hindi(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_japanese(text: str) -> str:
    text = text.replace(" ", "")
    return text.strip()

def normalize_korean(text: str) -> str:
    return text.strip()

def normalize_french(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_spanish(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_italian(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_portuguese(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_russian(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_generic(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def translate_text(text: str, src_lang_code: str = "auto", model_name_for_cache: str = None) -> str:
    """Translate text to Japanese using JANA-Light"""
    lid_model, translator_model, translator_tokenizer, _, _ = get_models()
    
    limiter = get_rate_limiter()
    if not limiter.allow():
        return "[Rate limit exceeded: slow down]"

    cached = cache_lookup(text, src_lang_code, model_name_for_cache)
    if cached:
        return cached

    if src_lang_code == "auto":
        predictions = lid_model.predict(text, k=1)
        src_lang_code = predictions[0][0].replace("__label__", "")
        conf = float(predictions[1][0])
        if conf < 0.5:
            src_lang_code = "en"  # fallback to English if uncertain
    else:
        conf = 1.0

    # Map to HF model language code (only base language, no script suffix)
    src_lang_hf = LANGUAGE_CODE_MAPPING.get(src_lang_code, src_lang_code)
    
    # Ensure a valid lightweight model name is always set
    model_name_for_cache = model_name_for_cache or "facebook/m2m100_418M"

    try:
        translator = pipeline(
            "translation",
            model=translator_model,
            tokenizer=translator_tokenizer,
            src_lang=src_lang_hf,
            tgt_lang="ja",
            device=0 if torch.cuda.is_available() else -1,
            max_length=2048,
            num_beams=5,
            no_repeat_ngram_size=3
        )

        result = translator(text)[0]['translation_text']

        # Apply language-specific post-processing
        if src_lang_hf == "hi":
            result = normalize_hindi(result)
        elif src_lang_hf == "ja":
            result = normalize_japanese(result)
        elif src_lang_hf == "ko":
            result = normalize_korean(result)
        elif src_lang_hf == "fr":
            result = normalize_french(result)
        elif src_lang_hf == "es":
            result = normalize_spanish(result)
        elif src_lang_hf == "it":
            result = normalize_italian(result)
        elif src_lang_hf == "pt":
            result = normalize_portuguese(result)
        elif src_lang_hf == "ru":
            result = normalize_russian(result)
        else:
            result = normalize_generic(result)

        # Existing Japanese post-processing
        result = post_process_japanese(result)

        if not is_japanese(result):
            result = "[NOT JAPANESE OUTPUT] " + result

        cache_store(text, src_lang_code, model_name_for_cache, result)
        return result

    except Exception as e:
        return f"[Translation error: {str(e)}]"
