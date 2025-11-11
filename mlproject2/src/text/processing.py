import re
from typing import List

# simple alphanumeric tokenizer
_ALNUM = re.compile(r"[A-Za-z0-9_]+")

# compact stopword set (you can edit)
_STOPWORDS = set("""a an and are as at be by for from has have he her hers him his i in is it its of on or our she that the their them they this to was were what when where which who will with you your""".split())

def _tokenize_raw(text: str) -> List[str]:
    """Return lowercase alphanumeric tokens."""
    return _ALNUM.findall(text.lower())

def prepare_text(text: str, remove_stop: bool = True) -> List[str]:
    """Primary preprocessing function used by the reworked project."""
    toks = _tokenize_raw(text)
    if remove_stop:
        toks = [t for t in toks if t not in _STOPWORDS and not t.isdigit()]
    return toks

# Compatibility wrapper: main.py expects preprocess()
def preprocess(text: str, remove_stop: bool = True) -> List[str]:
    """
    Compatibility wrapper so older callers (`preprocess`) continue to work.
    Delegates to prepare_text.
    """
    return prepare_text(text, remove_stop)
