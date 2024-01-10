# Construct the inverted index 
import os
import re
import json
from preprocessing import *
import math
from collections import Counter

def get_cached_tokens():
    with open("./cached/tokens.json", "r") as f:
        tokens = json.load(f)
    return tokens

# Extract the docNO for a document 
def extract_document_number(document): 
    pattern = re.compile(r'<DOCNO>(.*?)</DOCNO>', re.DOTALL)
    matches = pattern.findall(document)
    text = ' '.join(matches)
    return text

# Returns a list with all the words in the document (without stopwords but still containing duplicates)
def process_document(string_doc):
    text_without_numbers = re.sub(r'\d+', '', string_doc)
    text_stemmed = stemSentence(text_without_numbers) # stem the text
    tokenized_text = tokenize_string(text_stemmed)
    tokenized_text_lower = []
    for x in tokenized_text : 
        tokenized_text_lower.append(x.lower())
    return_value = remove_stopwords(tokenized_text_lower)
    return return_value

# reads the file, split in document , creates a dictionnary -> {document id , document content string}
def map_documents(file_name): 
    file_content = read_file(file_name)
    #split file in document 
    splitted_file = file_content.split('</DOC>')
    docs = {}
    documents=[] # stores all document numbers 
    for doc in splitted_file: 
        document_number = extract_document_number(doc)
        document_number = document_number.replace(" ", "") #remove white space from DOCNO
        documents.append(document_number)
        extracted_text = extract_text(doc)
        processed_document = process_document( extracted_text )
        docs[document_number] = processed_document
    return docs, documents

# print(process_files(files_names))
def produce_index (files_names): 
    # Get list of unique terms (list of tokens)
    tokens = get_cached_tokens()
    
    # Initializing the inverted index dict, documents and max_frequency
    documents=[]
    max_frequency= {} #used to find the token that has the maximum frequency across all documents

    inverted_index = {}
    for token in tokens:
        inverted_index[token] = {}

    # Add tokens as keys for the inverted index 
    for file in files_names : 
        mapped_documents,documentNo = map_documents(file)
        documents = documents + documentNo
        for key in mapped_documents : 
            if key == "" or key == None or len(mapped_documents[key]) == 0: 
                continue
            word_counts = Counter(mapped_documents[key])
            most_common = word_counts.most_common(1)
            max_frequency[key] =  most_common[0][1]
            for term in mapped_documents[key]: 
                if term in inverted_index : 
                    inverted_index[term][key] = mapped_documents[key].count(term)
    return inverted_index, documents, max_frequency
   


