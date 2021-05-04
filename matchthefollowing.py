import json
import requests
import string
import re
import nltk
import string
import itertools
import streamlit as st
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
import pke
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import traceback
from flashtext import KeywordProcessor
import re
from pprint import pprint
import random
import pandas as pd
# from prettytable import PrettyTable
# from IPython.display import Markdown, display

def file_selector_match():
    vAR_file = st.file_uploader('Upload the text file',type=['txt'],key='3')
    if vAR_file is not None:
        vAR_text = vAR_file.read().decode("utf-8")
        st.write('File Content: '+ vAR_text)
        return vAR_text

@st.cache(show_spinner=False)
def get_keywords(text):
    vAR_out=[]
    try:
        vAR_extractor = pke.unsupervised.YAKE()
        vAR_extractor.load_document(input=text)
        # pos = {'VERB', 'ADJ', 'NOUN'}
        vAR_pos ={'NOUN'}
        vAR_stoplist = list(string.punctuation)
        vAR_stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        vAR_stoplist += stopwords.words('english')
        vAR_extractor.candidate_selection(n=2,pos=vAR_pos, stoplist=vAR_stoplist)

        vAR_extractor.candidate_weighting(window=3,
                                      stoplist=vAR_stoplist,
                                      use_stems=False)

        vAR_keyphrases = vAR_extractor.get_n_best(n=30)
        
        for val in vAR_keyphrases:
            vAR_out.append(val[0])
    except:
        vAR_out = []
        traceback.print_exc()

    return vAR_out

#Extract sentences having the keywords that is extracted before.
@st.cache(show_spinner=False)
def get_sentences_for_keyword(keywords, sentences):
    vAR_keyword_processor = KeywordProcessor()
    vAR_keyword_sentences = {}
    for word in keywords:
        vAR_keyword_sentences[word] = []
        vAR_keyword_processor.add_keyword(word)
    for sentence in sentences:
        vAR_keywords_found = vAR_keyword_processor.extract_keywords(sentence)
        for key in vAR_keywords_found:
            vAR_keyword_sentences[key].append(sentence)

    for key in vAR_keyword_sentences.keys():
        vAR_values = vAR_keyword_sentences[key]
        vAR_values = sorted(vAR_values, key=len, reverse=False)
        vAR_keyword_sentences[key] = vAR_values
    return vAR_keyword_sentences


def sentence_answers(keyword_sentence_mapping):
    vAR_answers = []
    vAR_final_sentences = []
    for k,v in keyword_sentence_mapping.items():
        if len(v)>0:
            vAR_match = v[0].lower()
            vAR_answers.append(k)
            if k in vAR_match:
                vAR_temp = re.compile(re.escape(k), re.IGNORECASE)
                vAR_final_sentences.append(vAR_temp.sub('<answer>',vAR_match))
            else:
                vAR_final_sentences.append(vAR_match)
    return vAR_final_sentences, vAR_answers

# def printmd(string):
#     display(Markdown(string))

@st.cache(show_spinner=False)
@st.cache(allow_output_mutation=True)
def question(keyword_sentence_mapping):
    # tab = PrettyTable()
    vAR_final_sentences, vAR_answers  = sentence_answers(keyword_sentence_mapping)
    random.shuffle(vAR_answers)
    random.shuffle(vAR_final_sentences)
    vAR_cols_dict = {
        "A": vAR_answers,
        "B": vAR_final_sentences
    }
    pd.set_option("display.max_colwidth", None)
    vAR_cols = pd.DataFrame(vAR_cols_dict)
    return vAR_cols
    # tab.field_names=['A', 'B']
    # tab.align["A"] = "l"
    # tab.align["B"] = "l"

    # # printmd('**Match column A with column B**')
    # for word,context in zip(answers,final_sentences):
    #     tab.add_row([word,context.replace("\n"," ")])
    #     tab.add_row(['',''])
#     return tab
