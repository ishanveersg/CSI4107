from tqdm import tqdm
import os
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def read_file(file_name): 
    f = open(file_name,"r")
    text = f.read()
    return text 

def merge_files(folder_path: str, output_file: str):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as infile:
                    outfile.write(infile.read())

def read_initial_results(file_name):
    result = {}
    with open(file_name, 'r') as f:
        for line in tqdm(f, desc='Reading initial files'):
            fields = line.strip().split()
            topic_id = fields[0]
            docno = fields[2]
            if topic_id not in result:
                result[topic_id] = []
            if len(result[topic_id]) >= 1000:
                continue
            result[topic_id].append(docno)
            
    # write initial results to file
    with open("./cached/initial_results.json", "w") as file:
                json.dump(result, file)

def extract_docs(file_content):
    docs = re.findall(r'<DOC>(.+?)</DOC>', file_content, flags=re.DOTALL)
    doc_text = []
    for doc in tqdm(docs, desc='Extracting docs'):
        match = re.search(r'<TEXT>\s*(.+?)\s*</TEXT>', doc, flags=re.DOTALL)
        if match:
            doc_text.append(match.group(1))
        else:
            doc_text.append('')
    docnos = [re.search(r'<DOCNO>\s*(.+?)\s*</DOCNO>', doc).group(1) for doc in docs]
    doc_text = [' '.join(re.findall(r'\b[a-zA-Z]+\b', t)) for t in doc_text]
    doc_dict = {docnos[i]: doc_text[i] for i in range(len(docnos))}
    return doc_dict 

def extract_query(query):
    query_pattern = re.compile(r'<title>(.*?)<desc>(.*?)<narr>(.*?)</top>', re.DOTALL)
    matches = query_pattern.findall(query)
    results = {}
    count=1
    for match in tqdm(matches, desc='Extracting queries'):
        result = match[0].strip() + ' ' + match[1].strip()  # combine the two extracted strings with a space separator
        result = result.replace('\n', ' ')  # remove \n characters from the result
        results[count]= result
        count+=1
    return results

def embed_text(text_list, textNo):
    bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    text_embedded = bert_model.encode(text_list, batch_size = 500, show_progress_bar = True)
    text_embedded= text_embedded.tolist()
    text_dict = {textNo[i]: text_embedded[i] for i in range(len(textNo))}
    return text_dict


# Using 1000 first docs 
def calculate_cosine_similarity(queries_embedded, docs_embedded, initial_results_dict):   
    all_results = []
    for queryNo, docNo_list in initial_results_dict.items():
        query_embedding = queries_embedded[queryNo] 
        results = []
        for docNo in tqdm(docNo_list,total=len(initial_results_dict[queryNo]), desc='Calculating cosine similarity'):
            doc_embedding = docs_embedded[docNo]
            similarity = cosine_similarity([query_embedding], [doc_embedding])
            results.append((docNo, similarity.item()))
        results.sort(key=lambda x: x[1], reverse=True)
        all_results.append(results)
    return all_results

def write_to_file(results):
    with open("./cached/Results_1000.txt", "w") as f:
        for i, results in enumerate(results):
            for rank, (docno, score) in enumerate(results):
                f.write(f"{i+1} Q0 {docno} {rank+1} {score} run_name\n")
                
def preprocessing():
    
    if not os.path.exists("corpus.txt"):
        # Merges all documents from `coll` folder into a single file
        merge_files("../coll", "corpus.txt")

    with open("corpus.txt") as f:
        file_content= f.read()
    
    with open("../queries") as f:
        queries = f.read()
    
    # Extract queries and documents 
    doc_dict = extract_docs(file_content)
    query_dict = extract_query(queries)

    # Write extracted queries and documents to file 
    with open("./cached/extracted_docs.json", "w") as file:
            json.dump(doc_dict, file)
    with open("./cached/extracted_queries.json", "w") as file:
            json.dump(query_dict, file)

    # Reading initial 
    # read_initial_results('Results.txt')

def get_embedding():
    # Load extracted queries and documents 
    with open("./cached/extracted_docs.json", "r") as f:
            doc_dict = json.load(f)
    with open("./cached/extracted_queries.json", "r") as f:
            query_dict = json.load(f)

    # embedding queries 
    queries_embedded = embed_text(list(query_dict.values()),list(query_dict.keys()))

    #Write embedded queries to file 
    with open("./cached/embedded_queries.json", "w") as file:
            json.dump(queries_embedded, file)

    # embedding documents
    docs_embedded = embed_text(list(doc_dict.values()),list(doc_dict.keys()))

    # Write embedded documents to file 
    with open("./cached/embedded_docs.json", "w") as file:
            json.dump(docs_embedded, file)

def main():
           
    preprocessing()
    get_embedding()

    print("Loading initial results")
    with open("./cached/initial_results.json", "r") as f:
            initial_results_dict= json.load(f)  
    print ("Initial results successfully loaded")    
    print ("-----------------------------------------------------------------------------------------------------")  

    # Load embedded queries
    print("Loading embedded queries")
    with open("./cached/embedded_queries.json", "r") as f:
            queries_embedded = json.load(f)
    print ("Embedded queries successfully loaded")
    print ("-----------------------------------------------------------------------------------------------------") 

    # Load embedded documents
    print ("Loading embedded documents")
    with open("./cached/embedded_docs.json", "r") as f:
            docs_embedded = json.load(f)
    print ("embedded documents successfully loaded")
    print ("-----------------------------------------------------------------------------------------------------") 

    print ("Calculating cosine similarity for 50 queries")
    results= calculate_cosine_similarity(queries_embedded, docs_embedded,initial_results_dict)
    print("Cosine Similarity for 50 queries done")
    print ("-----------------------------------------------------------------------------------------------------") 

    print("Writing results to file")
    write_to_file(results)
    print("Writing results to file done")


if __name__ == '__main__':
    main()