import altair as alt
from PIL import Image
import streamlit as st

from gensim.models import Word2Vec
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
import gensim
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def input_text(text1):
    print('content --------- ',text1)
    words = [word for word in text1.split() if word.lower() not in ENGLISH_STOP_WORDS]
    print("Below are the stopwords removed from the given text")
    print({word for word in text1.split() if word.lower() in ENGLISH_STOP_WORDS})
    text3 = " ".join(words)
    text2 = sent_tokenize(text3)
    clean_text = []
    for text in text2:
        clean_text.append(gensim.utils.simple_preprocess(text))
    #min_count = 1 changed from 5
    model = gensim.models.Word2Vec(clean_text, vector_size=300, window=5, min_count = 1, workers=10)
    model.train(clean_text, total_examples = len(clean_text), epochs=10)
    return model

def input_word(model,word):
    print('Top Similar words correspond to ', word, 'are: ')
    output = model.wv.most_similar(positive=word, topn = 10)
    return output