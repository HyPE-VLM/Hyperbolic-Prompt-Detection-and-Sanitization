import re

from hyps.prompt_sanitization.stopwords import get_english_stopwords
from hyps.prompt_sanitization.thesaurus_llm import (
    get_thesaurus_antonyms,
    choose_best_antonym,
    substitute_word,
)
from hyps.prompt_sanitization.word_removal import remove_word


_WORD_STRIP_RE = re.compile(r"^[^\w]+|[^\w]+$")


def _normalize_token(token: str) -> str:
    return _WORD_STRIP_RE.sub("", token.strip().lower())


def get_top_k_influential_words_no_stopwords(word_attributions, k=1):
    stop_words = get_english_stopwords()

    filtered = []
    for w, score in word_attributions:
        if score <= 0:
            continue
        nw = _normalize_token(w)
        if not nw:
            continue
        if nw in stop_words:
            continue
        filtered.append((w, score))

    if not filtered:
        return []

    filtered.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _s in filtered[:k]]


def process_prompt(harmful_prompt, word_attributions, k, model_predict_fn):
    """
    Substitute with thesaurus antonym if available, else remove the word.
    """
    top_harmful_words = get_top_k_influential_words_no_stopwords(word_attributions, k=k)

    result = {
        "original_prompt": harmful_prompt,
        "top_influential_words": top_harmful_words,
        "antonym_words": [],
        "removed_words": [],
    }

    new_prompt = harmful_prompt
    antonym_pairs = []
    removed = []

    for w in top_harmful_words:
        if not w:
            continue

        antonyms = get_thesaurus_antonyms(w.lower())
        if antonyms:
            chosen = choose_best_antonym(w.lower(), antonyms)
            antonym_pairs.append((w.lower(), chosen))
            new_prompt = substitute_word(new_prompt, w, chosen)
        else:
            removed.append(w)
            new_prompt = remove_word(new_prompt, w)

    result.update({
        "antonym_words": antonym_pairs,
        "removed_words": removed,
        "final_prompt": new_prompt,
    })

    prediction = model_predict_fn(new_prompt)
    final_pred = "malicious" if prediction[0].item() == 0 else "benign"
    result.update({"final_pred": final_pred})
    return result
