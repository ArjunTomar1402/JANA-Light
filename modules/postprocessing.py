import re

def clean_english(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_korean(text: str) -> str:
    # Remove stray Latin chars stuck to Hangul
    text = re.sub(r'([a-zA-Z])([\uAC00-\uD7AF])', r'\1 \2', text)
    text = re.sub(r'([\uAC00-\uD7AF])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_french(text: str) -> str:
    # Fix spaces before punctuation like ? ! :
    text = re.sub(r'\s+([?!:;])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_spanish(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_italian(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_portuguese(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_russian(text: str) -> str:
    # Remove weird Latin + Cyrillic mix
    text = re.sub(r'([a-zA-Z])([\u0400-\u04FF])', r'\1 \2', text)
    text = re.sub(r'([\u0400-\u04FF])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_japanese(text: str) -> str:
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)  # strip control chars
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace(" ?", "？").replace(" !", "！").replace(" :", "：")
    text = re.sub(r'^\[NOT JAPANESE OUTPUT\]\s*', '', text)
    return text

def clean_hindi(text: str) -> str:
    text = re.sub(r'([a-zA-Z])([\u0900-\u097F])', r'\1 \2', text)
    text = re.sub(r'([\u0900-\u097F])([a-zA-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Dispatcher
CLEANERS = {
    "en": clean_english,
    "ko": clean_korean,
    "fr": clean_french,
    "es": clean_spanish,
    "it": clean_italian,
    "pt": clean_portuguese,
    "ru": clean_russian,
    "ja": clean_japanese,
    "hi": clean_hindi,
}

def post_process(text: str, lang_code: str) -> str:
    if not isinstance(text, str):
        return text
    cleaner = CLEANERS.get(lang_code, lambda x: x)
    return cleaner(text)
