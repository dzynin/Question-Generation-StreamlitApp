############ Importing required libraries #################

import altair as alt
from PIL import Image
import streamlit as st
from gensim.models import Word2Vec
# from nltk import sent_tokenize
#from gensim.models.word2vec import LineSentence
import nltk
nltk.download('punkt')
import gensim
#from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import warnings
warnings.filterwarnings('ignore')
import re

########### Creating function to remove stopwords from the text ###############

def remove_stopwords(text1):
    words = [word for word in text1.split() if word.lower() not in ENGLISH_STOP_WORDS]
#    st.write("Below are the stopwords removed from the given text")
#    st.write({word for word in text1.split() if word.lower() in ENGLISH_STOP_WORDS})
    text3 = " ".join(words)
#    text2 = LineSentence(text3)
    return text3

########### Creating function to tokenize the text into sentences ###############

def sent_tokenize(text3):
#    if st.button('Click to Tokenize'):
#    text2 = sent_tokenize(text3)
    text2 = re.compile('[.!?] ').split(text3)
    return text2

########### Creating function to remove special characters and convert the text into lower case ###############

def remove_special_Charac(text2):
    clean_text = []
    for text in text2:
        clean_text.append(gensim.utils.simple_preprocess(text))
    return clean_text

########### Creating function to train the model using Word2Vec for the processed text ###############

def Train_Model(clean_text):
#    if st.button('Click to Train the model'):
    model = gensim.models.Word2Vec(clean_text, vector_size=300, window=5, min_count = 5, workers=10)
    model.train(clean_text, total_examples = len(clean_text), epochs=10)
    return model

########### Creating function to apply the trained model ###############

def Model_Outcome(model, word):
#    if st.button('Click to Train the model'):
    st.write('Top Similar words correspond to ', word, 'are: ')
    output = model.wv.most_similar(positive=word, topn = 10)
    return output
