from core.logging_config import get_logger

logger = get_logger(__name__)

_nltk_initialized = False

def _ensure_nltk() -> bool:
    global _nltk_initialized
    if _nltk_initialized:
        return True
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        _nltk_initialized = True
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize NLTK: {e}")
        return False


def curate_text(text: str) -> str:
    if not _ensure_nltk():
        return text
    
    try:
        from nltk.tokenize import sent_tokenize
        sentences = sent_tokenize(text)
        return ' '.join(sentences)
    except Exception as e:
        logger.warning(f"Text curation failed: {e}")
        return text