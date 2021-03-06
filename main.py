import streamlit as st
import traceback
from PIL import Image
# Utils Pkgs
import textwrap
import pandas as pd
import base64
from datetime import datetime
from nltk.tokenize import sent_tokenize
import spacy
from tabulate import tabulate
from streamlit import caching
import SessionState


spacy.load("en_core_web_sm")

from true_false import  pos_tree_from_sentence,get_np_vp,alternate_sentences,summarize_text,fuzzy_dup_remove,file_selector_tf
from fill_blank import get_noun_adj_verb,get_sentences_for_keyword,get_fill_in_the_blanks,file_selector_fill
from matchthefollowing import  get_keywords,get_sentences_for_keyword,question,file_selector_match
from mcq import get_keywords, get_sentences_for_keyword, kw_distractors, getMCQ,file_selector_mcq
from word_sim import remove_stopwords,sent_tokenize, remove_special_Charac, Train_Model, Model_Outcome


def file_selector():
    file = st.file_uploader('Upload the text file',type=['txt'])
    if file is not None:
        text = file.read().decode("utf-8")
        st.write('File Content: '+ text)
        return text

def dtime():
      return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    
# Tokenizing sentence using nltk sent_tokenize
@st.cache(show_spinner=False)
def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences    
    
def output_file(out, quest_type):
      with open("output.txt","a",encoding="utf-8") as f:
        if quest_type == "Input Text":
              f.write("="*100+"\n")
        else:
              f.write("-"*100+"\n")
        dt = dtime()
        f.write(f"{dt} {quest_type}:\n")
        f.write("-"*100+"\n\n")
        if quest_type == "Match the Following":
            f.write(tabulate(out,showindex=False, headers=out.columns, tablefmt="grid"))
            # out.to_string(f,index = False)
            f.write("\n")
        elif quest_type == "Input Text":
            f.write(f"{out}")
            f.write("\n")
        elif quest_type == "Fill in The Blanks":
            for i,sent in enumerate(out["sentences"]):
                f.write(f"{str(i+1)}. {sent}\n")
            f.write("\n"+str(out["keys"])+"\n")
        elif quest_type == "MCQ":
            count = 1
            for quest,options in out.items():
                asci = 97
                f.write(f"{str(count)}. {quest}")
                if options:
                    for opt in options:
                        f.write(chr(asci)+")"+" "+opt.title()+"\r\n")
                        asci += 1
                    f.write("\r\n")
                count += 1
        else:
            for i,que in enumerate(out):
                f.write(f"{str(i+1)}. {que}\r\n")
        f.write("\r\n")
        
             
def download_link(object_to_download, download_filename, download_link_text,quest_type):
    """
    Generates a link to download the given object_to_download.
    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.
    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!','required_question_type')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!','required_question_type')
    """
#     if isinstance(object_to_download,pd.DataFrame):
#         object_to_download = object_to_download.to_csv(index=False)
#     object_to_download = str(object_to_download)
    object_to_download1 = ""
    if quest_type == "Input Text":
        object_to_download1 = "="*100+"\r\n"
    else:
        object_to_download1 = "-"*100+"\r\n"
    dt = dtime()
    object_to_download1+=f"{dt} {quest_type}:\r\n"
    object_to_download1+="-"*100+"\r\n\r\n"
    if quest_type == "Match the Following":
        # object_to_download1+=f"{str(object_to_download)}\r\n"
        object_to_download1+=tabulate(object_to_download,showindex=False,\
             headers=object_to_download.columns,tablefmt="grid")
    elif quest_type == "Input Text":
            object_to_download1+=f"{object_to_download}"
    elif quest_type == "Fill in The Blanks":
        for i,sent in enumerate(object_to_download["sentences"]):
            object_to_download1+=f"{str(i+1)}. {sent}\r\n"
        object_to_download1+="\n"+str(object_to_download["keys"])+"\r\n"
    elif quest_type == "MCQ":
        count = 1
        for quest,options in object_to_download.items():
            asci = 97
            object_to_download1+=f"{str(count)}. {quest}\n"
            if options:
                for opt in options:
                    object_to_download1+=chr(asci)+")"+" "+opt.title()+"\n"
                    asci += 1
                object_to_download1+=f"\r\n"
            count += 1
    else:
        for i,que in enumerate(object_to_download):
            object_to_download1+=f"{str(i+1)}. {que}\r\n"
    object_to_download1+=f"\r\n"
    b64 = base64.b64encode(object_to_download1.encode()).decode()
    
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'
    
def match_the_foll():
    text = file_selector_match()
    quest = "Match the Following"
    ts_col1,ts_col2,ts_col3 = st.beta_columns((1,1,2))
    ts_col1.success("Run Model")
    ts_col2.success("Step 1")
    if ts_col3.button('Tokenize sentences'):
        if text is not None:
            with st.spinner("Processing input to tokenize sentence"):
                sentences = tokenize_sentences(text)
                st.write(sentences)
            st.success('Tokenizing completed ')
        else:
            st.error("Please select input file!")
    ek_col1,ek_col2,ek_col3 = st.beta_columns((1,1,2))
    ek_col1.success("Run Model")
    ek_col2.success("Step 2")
    if ek_col3.button('Extract Keywords'):
        if text is not None:
            with st.spinner("Processing input to extract keywords"):
                keywords = get_keywords(text)[:6]
                st.write(keywords)
            st.success('Keywords Extracted')
        else:
            st.error("Please select input file!")
    km_col1,km_col2,km_col3 = st.beta_columns((1,1,2))
    km_col1.success("Run Model")
    km_col2.success("Step 3")
    if km_col3.button('Sentence Keyword Match'):
        if text is not None:
            with st.spinner("Processing input to match keywords with sentences"):
                sentences = tokenize_sentences(text)
                keywords = get_keywords(text)[:6]
                keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
                st.write(keyword_sentence_mapping)
            st.success('Sentence Keyword Match Completed')
        else:
            st.error("Please select input file!")
    fq_col1,fq_col2,fq_col3 = st.beta_columns((1,1,2))
    fq_col1.success("Run Model")
    fq_col2.success("Step 4")
    if fq_col3.button('Match the Following Questions'):
        if text is not None:
            with st.spinner("Processing input to generate questions"):
                sentences = tokenize_sentences(text)
                keywords = get_keywords(text)[:6]
                keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
                mtf_table= question(keyword_sentence_mapping)
                # st.write(str(mtf_table))
                st.table(mtf_table)
                output_file(text,"Input Text")
                output_file(mtf_table, quest)
        else:
            st.error("Please select input file!")
    vm_col1,vm_col2,vm_col3 = st.beta_columns((1,1,2))
    vm_col1.success("Validate Model")
    vm_col2.success("Step 5")
    if vm_col3.button('View Model Outcome'):
        if text is not None:
            sentences = tokenize_sentences(text)
            keywords = get_keywords(text)[:6]
            keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
            mtf_table = question(keyword_sentence_mapping)
            st.markdown(download_link(mtf_table, 'model_output.txt', 'Click here to download your output!',quest),unsafe_allow_html=True)
        else:
            st.error("Please select input file!")

def mcq():
    text = file_selector_mcq()
    quest = "MCQ"
    ts_col1,ts_col2,ts_col3 = st.beta_columns((1,1,2))
    ts_col1.success("Run Model")
    ts_col2.success("Step 1")
    if ts_col3.button('Tokenize sentences'):
        if text is not None:
            with st.spinner("Processing input to tokenize sentence"):
                sentences = tokenize_sentences(text)
                st.write(sentences)
            st.success('Tokenizing completed ')
        else:
            st.error("Please select input file!")
    ek_col1,ek_col2,ek_col3 = st.beta_columns((1,1,2))
    ek_col1.success("Run Model")
    ek_col2.success("Step 2")
    if ek_col3.button('Extract Keywords'):
        if text is not None:
            with st.spinner("Processing input to extract keywords"):
                keywords = get_keywords(text)[:6]
                st.write(keywords)
            st.success('Keywords Extracted')
        else:
            st.error("Please select input file!")
    km_col1,km_col2,km_col3 = st.beta_columns((1,1,2))
    km_col1.success("Run Model")
    km_col2.success("Step 3")
    if km_col3.button('Sentence Keyword Match'):
        if text is not None:
            with st.spinner("Processing input to match keywords with sentences"):
                sentences = tokenize_sentences(text)
                keywords = get_keywords(text)[:6]
                keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
                st.write(keyword_sentence_mapping)
            st.success('Sentence Keyword Match Completed')
        else:
            st.error("Please select input file!")
    fq_col1,fq_col2,fq_col3 = st.beta_columns((1,1,2))
    fq_col1.success("Run Model")
    fq_col2.success("Step 4")
    if fq_col3.button('Multiple Choice Questions'):
        if text is not None:
            with st.spinner("Processing input to generate questions"):
                sentences = tokenize_sentences(text)
                keywords = get_keywords(text)[:6]
                keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
                choices = kw_distractors(keywords)
                mcq_ques = getMCQ(keyword_sentence_mapping,choices)
                st.write(mcq_ques)
                output_file(text,"Input Text")
                output_file(mcq_ques, quest)
        else:
            st.error("Please select input file!")
    vm_col1,vm_col2,vm_col3 = st.beta_columns((1,1,2))
    vm_col1.success("Validate Model")
    vm_col2.success("Step 5")
    if vm_col3.button('View Model Outcome'):
        if text is not None:
            sentences = tokenize_sentences(text)
            keywords = get_keywords(text)[:6]
            keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
            choices = kw_distractors(keywords)
            mcq_ques = getMCQ(keyword_sentence_mapping,choices)
            st.markdown(download_link(mcq_ques, 'model_output.txt', 'Click here to download your output!',quest),unsafe_allow_html=True)
        else:
            st.error("Please select input file!")


def fill_blank(sentence,noun_verbs_adj,keyword_sentence_mapping_noun_verbs_adj):
    text = file_selector_fill()
    quest = "Fill in The Blanks"
    ts_col1,ts_col2,ts_col3 = st.beta_columns((1,1,2))
    ts_col1.success("Run Model")
    ts_col2.success("Step 1")
    if ts_col3.button('Tokenize sentences'):
        if text is not None:
            with st.spinner("Processing input to tokenize sentence"):
                sentences = tokenize_sentences(text)
                st.write(sentences)
            st.success('Tokenizing completed ')
        else:
            st.error("Please select input file!")
    ek_col1,ek_col2,ek_col3 = st.beta_columns((1,1,2))
    ek_col1.success("Run Model")
    ek_col2.success("Step 2")
    if ek_col3.button('Extract Keywords'):
        if text is not None:
            with st.spinner("Processing input to extract keywords"):
                noun_verbs_adj = get_noun_adj_verb(text)
                st.write(noun_verbs_adj)
            st.success('Keywords Extracted')
        else:
            st.error("Please select input file!")
    sk_col1,sk_col2,sk_col3 = st.beta_columns((1,1,2))
    sk_col1.success("Run Model")
    sk_col2.success("Step 3")
    if sk_col3.button('Sentence Keyword Match'):
        if text is not None:
            with st.spinner("Processing input to match keywords with sentences"):
                sentences = tokenize_sentences(text)
                noun_verbs_adj = get_noun_adj_verb(text)
                keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
                st.write(keyword_sentence_mapping_noun_verbs_adj)
            st.success('Sentence Keyword Match Completed')
        else:
            st.error("Please select input file!")
    fb_col1,fb_col2,fb_col3 = st.beta_columns((1,1,2))
    fb_col1.success("Run Model")
    fb_col2.success("Step 4")
    if fb_col3.button('Fill in the Blank Questions'):
        if text is not None:
            with st.spinner("Processing input to generate Fill in the blank questions"):
                sentences = tokenize_sentences(text)
                noun_verbs_adj = get_noun_adj_verb(text)
                keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
                fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
                st.write(fill_in_the_blanks)
                output_file(text,"Input Text")
                output_file(fill_in_the_blanks, quest)
        else:
            st.error("Please select input file!")
    vm_col1,vm_col2,vm_col3 = st.beta_columns((1,1,2))
    vm_col1.success("Validate Model")
    vm_col2.success("Step 5")
    if vm_col3.button('View Model Outcome'):
        if text is not None:
            sentences = tokenize_sentences(text)
            noun_verbs_adj = get_noun_adj_verb(text)
            keyword_sentence_mapping_noun_verbs_adj = get_sentences_for_keyword(noun_verbs_adj, sentences)
            fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping_noun_verbs_adj)
            st.markdown(download_link(fill_in_the_blanks, 'model_output.txt', 'Click here to download your output!',quest),unsafe_allow_html=True)
        else:
            st.error("Please select input file!")

                
def true_false():
    text = file_selector_tf()
    quest = "True or False"
    st_col1,st_col2,st_col3 = st.beta_columns((1,1,2))
    st_col1.success("Run Model")
    st_col2.success("Step 1")
    if st_col3.button('Summarize Input'):
        if text is not None:
            with st.spinner("Summarizing input text based on weighted frequency"):
                sentences = summarize_text(text)
                st.write(sentences)
            st.success('Generated summarized sentences from given input')
        else:
            st.error("Please select input file!")
    ts_col1,ts_col2,ts_col3 = st.beta_columns((1,1,2))
    ts_col1.success("Run Model")
    ts_col2.success("Step 2")
    if ts_col3.button('Tokenize sentences'):
        if text is not None:
            with st.spinner("Processing input to tokenize sentence and get 1st sentence to generate question"):
                sentences = summarize_text(text)
                sentences = [i for n, i in enumerate(tokenize_sentences(sentences)) if i not in tokenize_sentences(sentences)[:n]]
                sentences = fuzzy_dup_remove(sentences)
                st.write(sentences)
            st.success('Tokenization completed')
        else:
            st.error("Please select input file!")
    wc_col1,wc_col2,wc_col3 = st.beta_columns((1,1,2))
    wc_col1.success("Run Model")
    wc_col2.success("Step 3")
    if wc_col3.button('Words Construction'):
        if text is not None:
            with st.spinner("Parsing input to construct words"):
                sentences = summarize_text(text)
                sentences = [i for n, i in enumerate(tokenize_sentences(sentences)) if i not in tokenize_sentences(sentences)[:n]][0]
                pos = pos_tree_from_sentence(sentences)
                st.write(pos)
            st.success('Sample Grammatical parsing completed')
        else:
            st.error("Please select input file!")
    sc_col1,sc_col2,sc_col3 = st.beta_columns((1,1,2))
    sc_col1.success("Run Model")
    sc_col2.success("Step 4")
    if sc_col3.button('Sentence Construction'):
        if text is not None:
            split_list = []
            with st.spinner("Splitting sentence in-progress"):
                sentences = summarize_text(text)
                sentences = [i for n, i in enumerate(tokenize_sentences(sentences)) if i not in tokenize_sentences(sentences)[:n]]
                sentences = fuzzy_dup_remove(sentences)
                for i,sentence in enumerate(sentences):
                    if i <5:
                        pos = pos_tree_from_sentence(sentence)
                        split_sentence = get_np_vp(pos,sentence)
                        split_list.append(split_sentence)
                print('split_sentence in app.py- ',split_list)
                st.write(split_list)
            st.success('Sentences are splitted')
        else:
            st.error("Please select input file!")
    as_col1,as_col2,as_col3 = st.beta_columns((1,1,2))
    as_col1.success("Run Model")
    as_col2.success("Step 5")
    if as_col3.button('Alternate Sentences'):
        if text is not None:
            alt_sent_list = []
            with st.spinner("Generating Alternate sentences"):
                sentences = summarize_text(text)
                sentences = [i for n, i in enumerate(tokenize_sentences(sentences)) if i not in tokenize_sentences(sentences)[:n]]
                sentences = fuzzy_dup_remove(sentences)
                for i,sentence in enumerate(sentences):
                    if i <5:
                        pos = pos_tree_from_sentence(sentence)
                        alt_sentence = alternate_sentences(pos,sentence)
                        alt_sent_list.append(alt_sentence)
                        flat_list = [item for sublist in alt_sent_list for item in sublist]
                st.write(flat_list)
                output_file(text,"Input Text")
                output_file(flat_list,quest)
        else:
            st.error("Please select input file!")
    vm_col1,vm_col2,vm_col3 = st.beta_columns((1,1,2))
    vm_col1.success("Validate Model")
    vm_col2.success("Step 6")
    if vm_col3.button('View Model Outcome'):
        if text is not None:
            alt_sent_list = []
            sentences = summarize_text(text)
            sentences = [i for n, i in enumerate(tokenize_sentences(sentences)) if i not in tokenize_sentences(sentences)[:n]]
            sentences = fuzzy_dup_remove(sentences)
            for i,sentence in enumerate(sentences):
                if i <5:
                    pos = pos_tree_from_sentence(sentence)
                    alt_sentence = alternate_sentences(pos,sentence)
                    alt_sent_list.append(alt_sentence)
                    flat_list = [item for sublist in alt_sent_list for item in sublist]
            st.markdown(download_link(flat_list, 'model_output.txt', 'Click here to download your output!',quest),unsafe_allow_html=True)
        else:
            st.error("Please select input file!")
            
            
            
            
            
            
            
def word_similarity():
    text = file_selector()
    if st.button("Remove Stopwords"):
        if text is not None:
            with st.spinner("In-Progress"):
                text3 = remove_stopwords(text)
            st.success("Successfully removed stopwords")
        else:
            st.error("Please select input file!")
            
    if st.button("Sentence Tokenization"):
        if text is not None:
            with st.spinner("In-Progress"):
                text3 = remove_stopwords(text)
                text2 = sent_tokenize(text3)
            st.success("Successfully tokenized the text into sentences")
        else:
            st.error("Please check the input file!")
            
    if st.button("Remove Special Characters and Convert the text to Lower Case"):
        if text is not None:
            with st.spinner("In-Progress"):
                text3 = remove_stopwords(text)
                text2 = sent_tokenize(text3)
                clean_text = remove_special_Charac(text2)
            st.success("Successfully removed special characters and converted the text to lower case")
        else:
            st.error("Please check the input file!")
            
    if st.button("Train the Model"):
        if text is not None:
            with st.spinner("In-Progress"):
                text3 = remove_stopwords(text)
                text2 = sent_tokenize(text3)
                clean_text = remove_special_Charac(text2)
                model = Train_Model(clean_text)
            st.success("Model training successful")
        else:
            st.error("Please check the input file!")
            
    word = st.text_input('Input your word here in lower case and press ENTER:')
    if len(word)>0:
        if st.button("Model Outcome"):
            if text is not None:
                with st.spinner("Applying the trained Model"):
                    text3 = remove_stopwords(text)
                    text2 = sent_tokenize(text3)
                    clean_text = remove_special_Charac(text2)
                    model = Train_Model(clean_text)
                    output = Model_Outcome(model,word)
                st.write(output)
            else:
                st.error("Please check the model!")
#    else:
#        st.error("Type some input word")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
def all_initialisations():
    local_css("style.css")
    image = Image.open('DeepSphere_Logo_Final.png')
    st.image(image)    
    st.markdown('<h2>NLP Simplifies Questions and Assignments Construction <br><font style="color: #5500FF;">Powered by Google Cloud & Colab</font></h2>',unsafe_allow_html=True)
    st.markdown('<hr style="border-top: 6px solid #8c8b8b; width: 150%;margin-left:-180px">',unsafe_allow_html=True)
    activities= ['Select Your Question Type','Fill in the Blanks','True or False', 'Match the Following', 'MCQ','Word Similarity']
    model_choices = ['Model Implemented','BERT']
    libraries = ['Library Used','spacy','nltk','tensorflow','allennlp','flashtext','streamlit','pke','sense2vec','gensim']
    gcp = ['GCP Services Used','VM Instance','Compute Engine']
    session = SessionState.get(run_id=0)
    reset = st.sidebar.button("Reset/Clear")
    if reset:
        session.run_id += 1
    choice = st.sidebar.selectbox('',activities,key=session.run_id)
    model_choice = st.sidebar.selectbox('',model_choices)
    libraries_choice = st.sidebar.selectbox('',libraries)
    gcp_services = st.sidebar.selectbox('',gcp)
    return choice
