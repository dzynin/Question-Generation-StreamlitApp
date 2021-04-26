import streamlit as st
import json
import requests
import string
import re
import nltk
import string
import itertools
import base64
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from pprint import pprint
# from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
import subprocess
from datetime import datetime


# keyword extraction using pke's MultipartiteRank Algorithm
# Working principle of MultipartiteRank Algorithm - https://www.aclweb.org/anthology/N18-2105.pdf
@st.cache(show_spinner=False)
def get_noun_adj_verb(text_content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text_content,language="en", max_length=10000000,
                        normalization='stemming')
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'VERB', 'ADJ', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos, stoplist=stoplist)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=30)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()

    return out
    

# Identify and matching sentence for each keyword
@st.cache(show_spinner=False)
def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences



# Handling case sensitive and removing duplicate keywords
@st.cache(show_spinner=False)
def get_fill_in_the_blanks(sentence_mapping):
    out={"title":"Fill in the blanks for these sentences with matching words at the top"}
    blank_sentences = []
    processed = []
    keys=[]
    for key in sentence_mapping:
        if len(sentence_mapping[key])>0:
            sent = sentence_mapping[key][0]
            # Compile a regular expression pattern into a regular expression object, which can be used for matching and other methods
            insensitive_sent = re.compile(re.escape(key), re.IGNORECASE)
            no_of_replacements =  len(re.findall(re.escape(key),sent,re.IGNORECASE))
            line = insensitive_sent.sub(' _________ ', sent)
            if (sentence_mapping[key][0] not in processed) and no_of_replacements<2:
                blank_sentences.append(line)
                processed.append(sentence_mapping[key][0])
                keys.append(key)
    out["sentences"]=blank_sentences[:10]
    out["keys"]=keys[:10]
    return out
