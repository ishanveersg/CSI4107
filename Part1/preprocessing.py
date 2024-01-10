import re
import string
import os
import json
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

def get_files(folder_path):
    files = os.listdir(folder_path) 
    return files

def read_file(file_name): 
    f = open(file_name,"r")
    text = f.read()
    return text 

def extract_text(document):
    pattern = re.compile(r'<TEXT>(.*?)</TEXT>', re.DOTALL)
    matches = pattern.findall(document)
    text = ' '.join(matches)
    translate_table = text.maketrans('', '', string.punctuation)
    no_punct = text.translate(translate_table)
    return no_punct

def tokenize_string(str): 
    return_value = [str.split()]
    return return_value[0]

def stemSentence(sentence):
    porter = PorterStemmer()
    token_words=  word_tokenize(sentence) # Using it here only as it doesn't produce the same exact result as my other function that I cannot use here
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return  "".join(stem_sentence)

def process_files (files):
    extracted_text = ""
    for file_name in files :  
        file_content = read_file(file_name)
        extracted_text += extract_text(file_content)
    # remove numbers
    text_without_numbers = re.sub(r'\d+', '', extracted_text)
    text_stemmed = stemSentence(text_without_numbers) # stem the text
    tokenized_text = tokenize_string(text_stemmed)
    tokenized_text_lower = list(map(lambda x: x.lower(), tokenized_text))  #convert to lower case 
    # remove duplicate words: https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    text_no_duplicate  = list(dict.fromkeys(tokenized_text_lower))
    #remove stop words
    return_value = remove_stopwords(text_no_duplicate)
    return return_value

def remove_stopwords(tokens): 
    with open("StopWords.txt") as f:
        stopwords= f.readlines()
    stopwords = [x.strip() for x in stopwords]
    final_tokens = []
    for token in tokens : 
        if token in stopwords: 
            continue 
        final_tokens.append(token)
    return final_tokens

def get_files_names(folder_path): 
    files = get_files(folder_path)
    files_names = list(map(lambda x: folder_path + "/" + x, files))
    return files_names

def get_all_files_tokens(folder_path): 
    files_names = get_files_names(folder_path)
    tokens = process_files(files_names)
    # print to a json file 
    with open("./cached/tokens.json", "w") as file:
        json.dump(tokens, file)
      


