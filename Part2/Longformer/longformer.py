import os
import re
from tqdm import tqdm
import torch
from transformers import LongformerTokenizer, LongformerModel

# Extracts description and title from a query file
def extract_query_desc_title(query):
    query_pattern = re.compile(r'<title>(.*?)<desc>(.*?)<narr>', re.DOTALL)
    matches = query_pattern.findall(query)
    results = []
    for match in matches:
        result = match[0].strip() + '. ' + match[1].strip()  # combine the two extracted strings with a period and space.
        result = result.replace('\n', '')  # remove newline characters
        results.append(result)
    return results

# Extracts title only from a query file
def extract_query_title(query):
    query_pattern = re.compile(r'<title>(.*?)<desc>', re.DOTALL)
    matches = query_pattern.findall(query)
    results = []
    for match in matches:
        result = match.strip()  # Remove leading and trailing whitespace
        result = result.replace('\n', ' ')  # Remove newline characters and replace them with spaces
        results.append(result)
    return results

def merge_files(folder_path: str, output_file: str):
    with open(output_file, 'w') as outfile:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as infile:
                    outfile.write(infile.read())

# Extracts the text from documents and their corresponding docnos
def extract_docs(file_content):
    docs = re.findall(r'<DOC>(.+?)</DOC>', file_content, flags=re.DOTALL)
    doc_text = []
    for doc in docs:
        match = re.search(r'<TEXT>\s*(.+?)\s*</TEXT>', doc, flags=re.DOTALL)
        if match:
            cleaned_text = match.group(1).replace('\n', ' ')  # Replace newline characters with spaces
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Replace any sequence of whitespace characters with a single space
            doc_text.append(cleaned_text)
        else:
            doc_text.append('')
    docnos = [re.search(r'<DOCNO>\s*(.+?)\s*</DOCNO>', doc).group(1) for doc in docs]
    return doc_text, docnos

# Tokenizes the text and returns the tokenized text
def tokenize(text, tokenizer, max_length=4096):
    tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt", max_length=max_length, truncation=True)
    return tokens

# Converts the tokenized text to embeddings
def convert_tokens_to_embeddings(tokenized_text, model):
    model.eval()
    with torch.no_grad():
        embeddings = model(tokenized_text)[0][:, 0, :].squeeze(0)
    return embeddings.cpu()

# Calculates the cosine similarity between the query and the documents
def calculate_cosine_similarity(query_embedding, doc_embeddings):
    cosine_sim = torch.nn.CosineSimilarity(dim=1)
    return cosine_sim(query_embedding.unsqueeze(0), doc_embeddings).cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("corpus.txt"):
        # Merges all documents from `coll` folder into a single file
        merge_files("../coll", "corpus.txt")

    with open("corpus.txt") as f:
        corpus = f.read()

    # Extracts the text and their corresponding docnos from documents 
    doc_text, docnos = extract_docs(corpus)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)

    tokenized_docs = [tokenize(text, tokenizer).to(device) for text in tqdm(doc_text, desc="Tokenizing documents")]
    doc_embeddings = torch.stack([convert_tokens_to_embeddings(token, model) for token in tqdm(tokenized_docs, desc="Calculating embeddings")])

    with open("../queries") as f:
        queries = f.read()

    # Extracts the description and title from queries
    query_strings = extract_query_desc_title(queries)

    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)

    with open("Results_longformer.txt", "w") as results_file:
            for query_id, query in tqdm(enumerate(query_strings, 1), total=len(query_strings), desc="Calculating cosine similarities"):
                tokenized_query = tokenize(query, tokenizer).to(device)
                query_embedding = convert_tokens_to_embeddings(tokenized_query, model).to(device)
            
                doc_embeddings = doc_embeddings.to(device) # Move doc_embeddings to GPU
                cosine_similarities = calculate_cosine_similarity(query_embedding, doc_embeddings) # Calculate cosine similarity on GPU
                doc_embeddings = doc_embeddings.cpu() # Move doc_embeddings back to CPU

                ranked_docs = sorted(zip(docnos, cosine_similarities), key=lambda x: x[1], reverse=True)#[:1000]

                for rank, (docno, score) in enumerate(ranked_docs, 1):
                    results_file.write(f"{query_id} Q0 {docno} {rank} {score.item()} longformer\n")

if __name__ == "__main__":
    main()
