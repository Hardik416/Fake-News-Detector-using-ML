"""
language_handler.py
-------------------
Detects input language and translates non-English text to English
before passing through the classification pipeline.

Supported:  English, Multiregional Languages of India
Requires:   pip install langdetect deep-translator
"""

from langdetect import detect, LangDetectException

try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("Warning: deep-translator not installed. Hindi translation disabled.")


SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'mr': 'Marathi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
}

# Languages we auto-translate to English before classification
TRANSLATE_TO_EN = {'hi', 'mr', 'bn', 'ta', 'te', 'gu', 'pa'}


def detect_language(text: str) -> dict:
    """
    Detect the language of the input text.

    Returns:
        dict with keys: code, name, confidence (approximate)
    """
    if not text or len(text.strip()) < 10:
        return {'code': 'en', 'name': 'English', 'confidence': 0.5}

    try:
        code = detect(text)
        name = SUPPORTED_LANGUAGES.get(code, code.upper())
        return {
            'code': code,
            'name': name,
            'confidence': 0.95,  # langdetect doesn't expose confidence directly
            'needs_translation': code in TRANSLATE_TO_EN
        }
    except LangDetectException:
        return {'code': 'en', 'name': 'English', 'confidence': 0.3, 'needs_translation': False}


def translate_to_english(text: str, source_lang: str = 'auto') -> dict:
    """
    Translate text to English using GoogleTranslator (free, no API key).

    Args:
        text:        Input text in any language
        source_lang: Language code (e.g. 'hi') or 'auto' for auto-detect

    Returns:
        dict with keys: original, translated, source_lang, success
    """
    if not TRANSLATOR_AVAILABLE:
        return {
            'original': text,
            'translated': text,
            'source_lang': source_lang,
            'success': False,
            'error': 'deep-translator not installed'
        }

    try:
        translator = GoogleTranslator(source=source_lang, target='en')
        # GoogleTranslator has a 5000 char limit per call
        if len(text) > 4900:
            # Chunk and translate
            chunks = [text[i:i+4900] for i in range(0, len(text), 4900)]
            translated_chunks = [translator.translate(chunk) for chunk in chunks]
            translated = ' '.join(translated_chunks)
        else:
            translated = translator.translate(text)

        return {
            'original': text,
            'translated': translated,
            'source_lang': source_lang,
            'success': True
        }
    except Exception as e:
        return {
            'original': text,
            'translated': text,  # fallback: use original
            'source_lang': source_lang,
            'success': False,
            'error': str(e)
        }


def prepare_text_for_classification(text: str) -> dict:
    """
    Full pipeline: detect language → translate if needed → return ready-to-classify text.

    Returns:
        dict with keys:
            - text_for_model:  cleaned, English text ready for classification
            - original_text:   original input
            - language:        detected language dict
            - was_translated:  bool
            - translation_info: translation result dict (if applicable)
    """
    lang_info = detect_language(text)

    result = {
        'original_text': text,
        'language': lang_info,
        'was_translated': False,
        'translation_info': None,
        'text_for_model': text
    }

    if lang_info.get('needs_translation', False):
        translation = translate_to_english(text, source_lang=lang_info['code'])
        result['was_translated'] = translation['success']
        result['translation_info'] = translation
        result['text_for_model'] = translation['translated'] if translation['success'] else text

    return result


# ─── Convenience wrapper ─────────────────────────────────────────────────────

def get_english_text(text: str) -> str:
    """Quick helper: return English version of any input text."""
    prepared = prepare_text_for_classification(text)
    return prepared['text_for_model']
