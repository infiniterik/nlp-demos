import streamlit as st

from nltk.stem.porter import *
from nltk.stem.snowball import *

p_stemmer = PorterStemmer()
s_stemmer = SnowballStemmer("english")

st.title("Stemming Demo")

word = st.text_input('Enter word to stem')
if word:
	p_stemmed = p_stemmer.stem(word)
	s_stemmed = s_stemmer.stem(word)
	st.title("Porter")
	st.write(word + '→'+ p_stemmed)
	st.title("Snowball")
	st.write(word + '→'+ s_stemmed)
