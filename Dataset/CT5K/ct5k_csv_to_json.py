# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import json
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

# %%
import nltk
# nltk.download('punkt')

# %%
import numpy as np
def find_from_to(post_toks, sentence_toks, noun_phrase_toks):
    sk = len(sentence_toks)
    nk = len(noun_phrase_toks)
    candidates = []
    for i in range(len(post_toks) - sk + 1):
        if tuple(post_toks[i:i+sk]) == tuple(sentence_toks):
            for j in range(i, i+sk-nk+1):
                if tuple(post_toks[j:j+nk]) == tuple(noun_phrase_toks):
                    candidates.append((i, i+nk))
            return candidates[np.random.randint(len(candidates))]
    raise ValueError("Noun phrase not found")


# %%
MAPPING = {"$AnswerA": "positive", "$AnswerB": "negative", "$AnswerC": "neutral"}

# %%
from typing import List, Dict
def csv_to_json(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    tk = SpacyTokenizer()
    result_list = []
    for i, row in df.iterrows():
        post = row["Post"]
        noun_phrase = row["NP"]
        sentence = row["Sentence"]
        token = [t.text for sent in nltk.sent_tokenize(post) for t in tk.tokenize(sent)]
        sent_tokens = [t.text for t in tk.tokenize(sentence)]
        np_token = [t.text for t in tk.tokenize(noun_phrase)]
        try:
            from_to = find_from_to(token, sent_tokens, np_token)
        except:
            print(row)
            print(token)
            print(sent_tokens)
            print(np_token)
            continue

        instance_dict = {
            "sentence": post,
            "token": token,
            "pos": None,
            "dependencies": None,
            "aspects": [{
                "term": np_token,
                "polarity": MAPPING[row["sentiment"]],
                "from": from_to[0],
                "to": from_to[1]
            }]
        }
        result_list.append(instance_dict)
    return result_list


# %%
with open("Test.json", "w") as f:
    json.dump(csv_to_json("CT5K_raw_test.csv"), f, indent=2)

with open("Train.json", "w") as f:
    json.dump(csv_to_json("CT5K_raw_train.csv"), f, indent=2)
