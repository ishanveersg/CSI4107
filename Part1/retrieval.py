from preprocessing import *
import math

# Extract title and descrition in a query 
def extract_query(query,count):
    pattern = re.compile(r'<title>(.*?)<narr>', re.DOTALL) # extract the content between title and narr
    matches = pattern.findall(query)
    to_stem = matches[count].replace("<desc>","") # remove desc tag
    translate_table = to_stem.maketrans('', '', string.punctuation)
    no_punct_to_stem = to_stem.translate(translate_table)
    first_query = stemSentence(no_punct_to_stem)
    first_query=first_query.split()
    final_query = []
    for x in first_query : 
        final_query.append(x.lower())
    no_stopwords_q= remove_stopwords(final_query)
    return no_stopwords_q

# Extract the title in a query
# def extract_query(query,count):
#     pattern = re.compile(r'<title>(.*?)<desc>', re.DOTALL)
#     matches = pattern.findall(query)
#     first_query = stemSentence(matches[count])
#     first_query=first_query.split()
#     final_query = []
#     for x in first_query : 
#         final_query.append(x.lower())
#     no_stopwords_q= remove_stopwords(final_query)
#     return no_stopwords_q

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

def compute_cossim_iq(query_files, idf_values,count,doc_tf_idf, doc_len) : 
    query= extract_query(query_files,count)  #returns a list of words for a given query
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
    
    print("-------- Reading files : STARTING ----------")
    # Read the list back from the file
    with open("./cached/inverted_index.json", "r") as f:
        inverted_index = json.load(f)

    with open("./cached/documentsNumbers.json", "r") as f:
        documentsNumbers = json.load(f)

    with open("./cached/documents_max_frequency.json", "r") as f:
        documents_max_frequency= json.load(f)
    
    # get_doc_tf_idf(inverted_index,documentsNumbers,documents_max_frequency) 
    
    with open("./cached/idf_values.json", "r") as f:
        idf_values = json.load(f)


    with open("./cached/document_tf_idf.json", "r") as f:
        doc_tf_idf = json.load(f)

    print("-------- Reading files : DONE ----------")
    print("-------- Calculate document length : STARTING ----------")
    # calculate document length 
    doc_len= doc_length(doc_tf_idf)

    print("-------- Calculate document length : DONE ----------")
    print("-------- Read query files : STARTING ----------")
    # Read test queries from file 
    query_files = read_file("test_query.txt")
    
    
    print("-------- Read query files : DONE ----------")
    # print("-------- Compute cosine similarity : STARTING ----------")
    # Compute cosine similarity for 1 query file 
    # results=compute_cossim_iq(query_files, idf_values, 0, doc_tf_idf, doc_len)
    # write_to_file((0+1),results)

    # print("-------- Compute cosine similarity : DONE ----------")
    print("-------- Compute cosine similarity for 50 queries : STARTING ----------")
    #CosSim for 50 queries 
    for i in range (0,50) :
        print("-------- Compute cosine similarity for QUERY " + str(i+1)+ " : STARTING ----------")
        results=compute_cossim_iq(query_files, idf_values, i, doc_tf_idf, doc_len)
        write_to_file((i+1),results)
        print("-------- Compute cosine similarity for QUERY " + str(i+1)+ " : DONE ----------")
        
    print("-------- Compute cosine similarity for 50 queries : DONE ----------")

if __name__ == '__main__':
    main()