import streamlit as st
from annotated_text import annotated_text

import time
import nltk
import spacy
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


st.title("Lesk Demo")
sentence = word_tokenize(st.sidebar.text_input("Input sentence here"))
target = st.sidebar.text_input("target word")
pos = st.sidebar.selectbox("part of speech", ("n", "v", "a", "r", "s"))
stop = st.sidebar.checkbox("Remove stopwords")
use_spacy = st.sidebar.checkbox("use word vectors")
stopword_list = stopwords.words("english")
colors = ["#D1FAFF", "#9BD1E5", "#6A8EAE", "#57A773", "#157145"]

@st.cache
def lesk(sentence, target, pos, use_spacy):
    if use_spacy:
        nlp = spacy.load("en_core_web_sm")
    overlaps = []
    synsets = wn.synsets(target, pos)
    scored = []
    for s in synsets:
        definition = word_tokenize(s.definition())
        if use_spacy:
            tspace = [nlp(d) for d in definition]
        score = []
        match = []
        for w in sentence:
            if stop and w in stopword_list:
                score.append(0)
                match.append("")
            elif use_spacy:
                tw = nlp(w)
                if tw[0].is_punct:
                    continue
                sim = 0
                tok = ""
                for t in range(len(definition)):
                    if tspace[t][0].is_punct:
                        continue
                    _sim = tw.similarity(tspace[t])
                    if _sim > sim:
                        tok = definition[t]
                        sim = _sim
                score.append(sim)
                match.append(tok)
            elif w in definition:
                score.append(1)
                match.append(w)
            else:
                score.append(0)
                match.append("")
        scored.append((score, s.name(), definition, match))
    return scored

def pick(val, min_v, max_v):
    c = int((len(colors)-1)*(val-min_v)/max_v)
    return colors[c]

def highlight_definition(definition, match, scores):
    result = ["Definition: "]
    weights = list(zip(match, scores))
    if use_spacy:
        fmt = "%.2f"
    else:
        fmt = "%d"
    color_range = [sum([x[0] for x in weights if d == x[1]]) for d in definition]
    color_range = [s for s in color_range if s]
    if not color_range:
        min_v = 0
        max_v = 1
    else:
        min_v = min(color_range)/2
        max_v = max(color_range)
    for d in definition:
        s = sum([x[0] for x in weights if d == x[1]])
        color = pick(s, min_v, max_v)
        if s:
            result.append((d, fmt % s, color))
        else:
            result.append(d+" ")
    return result

def highlight_sentence(sentence, scores, match):
    result = []
    if use_spacy:
        fmt = "%s, %.2f"
    else:
        fmt = "%s, %d"
    mscores = [s for s in scores if s]
    if not mscores:
        min_v = 0
        max_v = 1
    else:
        min_v = min(mscores)/2
        max_v = max(scores)
    for w, s, m in zip(sentence, scores, match):
        if s:
            color = pick(s, min_v, max_v)
            result.append((w, fmt%(m,s), color))
        else:
            result.append(w+" ")
    return result

if sentence and target:
    result = lesk(sentence, target, pos, use_spacy)
    mfs = True
    for winner in sorted(result, key=lambda x: -sum(x[0])):
        score = winner[1] + " was the most frequent sense. No overlap found."
        if sum(winner[0]) or not mfs:
            score = winner[1] + " had an overlap of " + str(sum(winner[0]))
        with st.expander(score):
            annotated_text(*highlight_definition(winner[2], winner[0], winner[3]))
            st.write("")
            annotated_text(*highlight_sentence(sentence, winner[0], winner[3]))
            st.write("")
        mfs = False
else:
    with st.spinner('Wait for it...'):
        time.sleep(5)