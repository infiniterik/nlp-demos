import streamlit as st

#@title Porter stemmer
import nltk

from nltk.stem.porter import *
p_stemmer = PorterStemmer()
st.title("Porter Stemmer")
word = st.text_input('Enter word to stem')
if word:
	st.title(word+'→'+p_stemmer.stem(word))
