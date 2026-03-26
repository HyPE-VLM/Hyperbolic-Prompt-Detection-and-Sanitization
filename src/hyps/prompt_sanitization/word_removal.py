import re

def remove_word(text: str, word: str) -> str:
    pattern = r"\b{}\b[,.!?;:]*\s*".format(re.escape(word))
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text


def get_top_k_influential_words(word_attributions, k=1):
    attributions = [t for t in word_attributions if t[1] > 0]
    if not attributions:
        return []
    attributions.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in attributions[:k]]


def process_prompt(harmful_prompt, word_attributions, k, model_predict_fn):
    top_harmful_words = get_top_k_influential_words(word_attributions, k=k)

    result = {
        "original_prompt": harmful_prompt,
        "top_influential_words": top_harmful_words,
        "removed_words": [],
    }

    new_prompt = harmful_prompt
    removed = []
    for w in top_harmful_words:
        if not w:
            continue
        removed.append(w)
        new_prompt = remove_word(new_prompt, w)

    result.update({
        "removed_words": removed,
        "final_prompt": new_prompt,
    })

    prediction = model_predict_fn(new_prompt)
    final_pred = "malicious" if prediction[0].item() == 0 else "benign"
    result.update({"final_pred": final_pred})
    return result
