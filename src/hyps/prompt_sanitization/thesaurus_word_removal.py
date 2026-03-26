from hyps.prompt_sanitization.thesaurus_llm import (
    get_thesaurus_antonyms,
    choose_best_antonym,
    substitute_word,
    get_top_k_influential_words,
)
from hyps.prompt_sanitization.word_removal import remove_word


def process_prompt(harmful_prompt, word_attributions, k, model_predict_fn):
    """
    For top-k influential words: substitute with thesaurus antonym if available, else remove the word.
    """
    top_harmful_words = get_top_k_influential_words(word_attributions, k=k)

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
