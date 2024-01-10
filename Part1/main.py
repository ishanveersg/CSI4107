# import 
import json
from preprocessing import get_all_files_tokens, get_files_names
from indexing import produce_index

# run preprocessing 



# produce the inverted index 


# Query preparation 


# query run 

print("--------STEP 1 : Preprocessing: STARTING ----------")

tokens =  get_all_files_tokens("./coll")

files_names = get_files_names("./coll")

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




