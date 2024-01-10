import re
import string
import os
import json
import math
from collections import Counter
import gensim.downloader as api
from tqdm import tqdm

model_fastText = api.load("fasttext-wiki-news-subwords-300")
print("=============||| fasttext-wiki-news-subwords-300 ")

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

def process_files (files):
    extracted_text = ""
    for file_name in files :  
        file_content = read_file(file_name)
        extracted_text += extract_text(file_content)
    # remove numbers
    text_without_numbers = re.sub(r'\d+', '', extracted_text)
    tokenized_text = tokenize_string(text_without_numbers)
    tokenized_text_lower = list(map(lambda x: x.lower(), tokenized_text))  #convert to lower case 
    # remove duplicate words: https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    text_no_duplicate  = list(dict.fromkeys(tokenized_text_lower))
    #remove stop words
    return_value = remove_stopwords(text_no_duplicate)
    return return_value

def remove_stopwords(tokens): 
    with open("../stopwords") as f:
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
    # text_stemmed = stemSentence(text_without_numbers) # stem the text
    tokenized_text = tokenize_string(text_without_numbers)
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
   
def expand_query(query, model, topn=5, similarity_threshold=0.6):
    tokens = query.split()
    expanded_query = []

    for token in tokens:
        expanded_query.append(token)
        
        if token in model:
            similar_words = model.most_similar(token, topn=topn)
            
            for word, similarity in similar_words:
                if similarity >= similarity_threshold:
                    expanded_query.append(word)
    
    return ' '.join(expanded_query)

# Extract title,descrition and narration in a query 
def extract_query(query,count):
    pattern = re.compile(r'<title>(.*?)</top>', re.DOTALL) # extract the content between title and narr
    matches = pattern.findall(query)
    to_stem = matches[count].replace("<desc>","") # remove desc tag
    to_stem = matches[count].replace("<narr>","") # remove narr tag
    translate_table = to_stem.maketrans('', '', string.punctuation)
    no_punct_to_stem = to_stem.translate(translate_table)
    no_punct_to_stem =  expand_query(no_punct_to_stem,model_fastText) # expand_string(no_punct_to_stem, word2vec_model)
    first_query = no_punct_to_stem 
    #stemSentence(no_punct_to_stem)
    first_query=first_query.split()
    final_query = []
    for x in first_query : 
        final_query.append(x.lower())
    no_stopwords_q= remove_stopwords(final_query)
    return no_stopwords_q

def query_tf(query):
    query_tf={}
    for token in query:
        if token in query_tf.keys():
            continue 
        query_tf[token]= query.count(token)
    return query_tf

def query_tf_idf(query_tf, idf_values):
    query_token={}
    for token in query_tf: 
        if token in idf_values.keys():
            query_token[token] = (query_tf[token] / max(query_tf.values())) * idf_values[token]
    return query_token

def query_length(query_tf_idf):
    length=0
    for value in query_tf_idf:
        length= length + query_tf_idf[value]**2
    length= math.sqrt(length)
    return length

def create_idf(inverted_index):
    numofDocuments = 79923
    idf_values = {}
    for key in inverted_index : 
        if (len(inverted_index[key]) == 0 ):
            idf_values[key] = 0
        else:
            idf_values[key] = math.log2(numofDocuments/ len(inverted_index[key]))
    return idf_values

def calculate_tf_idf(documents, inverted_index, idf_values, max_frequency):
    document_tf_idf={}
    for docNo in documents: 
        if docNo == "" or docNo == None:
             continue
        document_tf_idf[docNo]={}
        for token in inverted_index: 
            if (inverted_index.get(token,{}).get(docNo, None)) == None: 
                continue
            document_tf_idf[docNo][token]= (inverted_index[token][docNo] / max_frequency[docNo]) * idf_values[token]
    return document_tf_idf

def doc_length(doc_tf_idf):
    length_doc={}
    length=0
    for docNo in doc_tf_idf:
        for token in doc_tf_idf[docNo]:
            length= length + doc_tf_idf[docNo][token]**2
        length_doc[docNo]= math.sqrt(length)
        length=0
    return length_doc

def calc_cosSim(doc_tf_idf, q_tf_idf, doc_len, query_len):
    cosSim={}
    sum=0
    for docNo in doc_tf_idf: 
        for token in doc_tf_idf[docNo]:
            if token in q_tf_idf.keys():
                sum= sum+ (q_tf_idf[token] * doc_tf_idf[docNo][token])
        if sum != 0 :
            cosSim[docNo]= sum/ (doc_len[docNo] * query_len)
        sum=0
    return dict(sorted(cosSim.items(), key=lambda x: x[1], reverse=True))

def compute_cossim_iq(query, idf_values,doc_tf_idf, doc_len) : 
    q_tf= query_tf(query)                  #returns a list with tf for each word in a query 
    q_tf_idf=query_tf_idf(q_tf, idf_values) #calculates tf_idf of the query
    query_len= query_length(q_tf_idf)        #calculates the length of the query 
    results=calc_cosSim(doc_tf_idf, q_tf_idf, doc_len, query_len)  #calculates cosine Similarity with 1 query and the collection of documents (key: DocNo value: casine similarity)
    return results

def write_to_file(topicNo,results):
    count = 1
    for docNo in results:   
        to_print = str(topicNo) + " Q0 " + str(docNo) + " " + str(count) + " " + str(results[docNo]) + " " + "testTag"
        with open("Results.txt", "a") as file:
            print(to_print, file=file)
        count += 1 
        
# function to get the idf values and document tf-idf scores and store in separate files 
def get_doc_tf_idf(inverted_index,documentsNumbers,documents_max_frequency):
    print("-------- Calculate idf values : STARTING ----------")
    # calculate idf values 
    idf_values= create_idf(inverted_index)

    # Write the list to the file 
    with open("./cached/idf_values.json", "w") as file:
        json.dump(idf_values, file)
        
    print("-------- Calculate idf values : DONE ----------")
    print("-------- Calculate document tf idf values : STARTING ----------")
    #calculate document tf idf 
    doc_tf_idf=calculate_tf_idf(documentsNumbers, inverted_index,idf_values, documents_max_frequency)

    #Write document tf idf to the file 
    with open("./cached/document_tf_idf.json", "w") as file:
        json.dump(doc_tf_idf, file)

def main():
    
    print("--------STEP 1 : Preprocessing: STARTING ----------")

    get_all_files_tokens("../coll")

    files_names = get_files_names("../coll")

    print("--------STEP 1 : Preprocessing: DONE ----------")

    print("--------STEP 2 : Indexing and data storing : STARTING ----------")
    inverted_index, documentsNumbers, max_frequency = produce_index(files_names)

    with open("./cached/inverted_index.json", "w") as file:
        json.dump(inverted_index, file)

    with open("./cached/documents_max_frequency.json", "w") as file:
        json.dump(max_frequency, file)

    with open("./cached/documentsNumbers.json", "w") as file:
        json.dump(documentsNumbers, file)


    print("--------STEP 2 : Indexing and data storing : DONE ----------")
    
    print("-------- Reading files : STARTING ----------")
    # Read the list back from the file
    with open("./cached/inverted_index.json", "r") as f:
        inverted_index = json.load(f)

    with open("./cached/documentsNumbers.json", "r") as f:
        documentsNumbers = json.load(f)

    with open("./cached/documents_max_frequency.json", "r") as f:
        documents_max_frequency= json.load(f)
    
    get_doc_tf_idf(inverted_index,documentsNumbers,documents_max_frequency) 
    
    with open("./cached/idf_values.json", "r") as f:
        idf_values = json.load(f)


    with open("./cached/document_tf_idf.json", "r") as f:
        doc_tf_idf = json.load(f)

    print("-------- Reading files : DONE ----------")
    print("-------- Calculate document length : STARTING ----------")
    # calculate document length 
    doc_len= doc_length(doc_tf_idf)

    print("-------- Calculate document length : DONE ----------")
    print("-------- Load query files : STARTING ----------")
    # Read test queries from file 
    with open("../queries") as f:
        query_files = f.read()
        
    all_queries={}
    for i in tqdm(range(0, 50)):
        queries=extract_query(query_files,i)
        all_queries[i]=queries
    
    with open("./cached/queries.json", "w") as file:
        json.dump(all_queries, file)
    # load query files 
    with open("./cached/queries.json", "r") as f:
         query_files=json.load(f)
    
    print("-------- Load query files : DONE ----------")
  
    print("-------- Compute cosine similarity for 50 queries : STARTING ----------")
    
    #CosSim for 50 queries 
    for i, queryNo in tqdm(enumerate(query_files.keys()), total=len(query_files)) :
        print("-------- Compute cosine similarity for QUERY " + str(i+1)+ " : STARTING ----------")
        query= query_files[queryNo]
        results=compute_cossim_iq(query, idf_values, doc_tf_idf, doc_len)
        write_to_file((i+1),results)
        print("-------- Compute cosine similarity for QUERY " + str(i+1)+ " : DONE ----------")
        
    print("-------- Compute cosine similarity for 50 queries : DONE ----------")

if __name__ == '__main__':
    main()