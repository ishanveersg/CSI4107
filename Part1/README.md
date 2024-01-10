# CSI4107

## Group #20

| Student Name       | Student Number|
|--------------------|---------------|
| Andie Samadoulougou|300209487      | 
| Ishanveer Gobin    |300135454      |
| Kate Sin Yan Chun  |300144923      |

### How tasks were divided 

The three of us worked on the assignment all together during scheduled timeslots. 
Since all the steps were dependent on the preceding steps (i.e step 2 is dependent on step1 etc..), we thought it'll be better to work on it together so that we all understand the assignment in the same way. We practiced pair coding with a lead coder writing the structure of the program and the others creating the supporting functions and reviewing the code. We each took turns being the lead coder. 

## Note about the functionality of programs 

### Part1: Preprocessing 

For the preprocessing, we extracted the text from between the XML tags, tokenized it, remove stopwords and stemmed the resulting tokens.

**Functions: preprocessing.py**
* `get_files(folder_path)` takes in the folder path which contains the documents and returns all the document file names
* `read_file(file_name)` gets the file names using `get_files(folder_path)` function, opens the document file and reads the content into a string  
* `extract_text(documents)` takes in each document and extracts what is between the `<TEXT>` tags
* `tokenize_string(str)` converts the extracted text into a list of it's constituent words (list of tokens)
* `stemSentence(sentence)` stems a sentence using PorterStemmer from the Nltk library and returns it
* `process_files(files)` goes through all the documents, removes all numbers, all occurences of stop words(using the `remove_stopwords(tokens)` function) and all duplicates of the same word
* `remove_stopwords(tokens)` removes all stopwords from the tokens list 
* `get_files_names(folder_path)` gets a folder path and returns a list containing all the files names 
* `get_all_files_tokens(folder_path)` writes the token to a json file 

### Step2: Indexing (Inverted Index)

For the indexing, we used the tokens from the preprocessing step to create the inverted index. 

**Functions: indexing.py** 
* `get_cached_tokens()` retrieves the tokens from `Part1:Preprocessing` 
* `extract_document_number(document)` returns the document number of a document 
* `process_document(string_doc)` tokenizes the text, removes stops but keeps all duplicates
* `map_documents(file_name)`  split one file into individual documents and extracts the text between the `<TEXT>` tag. Returns the a dictionary with all tokens for each document and a ;ist of all documents numbers 
* `produce_index(files_names)` returns the inverted index, the documents numbers and the max frequencies for all documents after going through all the documents

### Part 3: Retrieval and Ranking

 In the retrieval step, we first calculated the document tf-idf and document lengths. Then, for each query, we extracted the title/title and description. Then we removed all punctuations, stemmed the tokens and removed all the stopwords.We then calculated the query length and tf-idf values and then computed the cosine similarity between the query and the collection of documents. For the 50 test queries, the results are stored in `Results.txt` sorted by descending order and grouped by query number.

**Functions: retrieval.py** 
* `extract_query(query,count)`  returns the tokens for a specific query. Note that there are 2 implementation of `extract_query(query,count)`. One to extract the tile only and one to extract the title and the description.
* `query_tf(query)` returns a list for the tf for each word in a query
* `query_tf_idf(query_tf,idf_values)` calculates the tf-idf for a query 
* `query_length(query_tf_idf)` calculates the length of a query 
* `create_idf(inverted_index)` takes in the inverted index and calculates the idf for each token using $log{_2}{(N/x)}$ where N is the total number of documents(79923) and $x$ is the document frequency
* `calculate_tf_idf(documents, inverted_index, idf_values, max_frequency)` calculates tf-idf for all documents 
* `doc_length(doc_tf_idf)` calculates the length of all documents 
* `calc_cosSim(doc_tf_idf, q_tf_idf, doc_len, query_len)` calculates the cosine similarity for each document with 1 query 
* `compute_cossim_iq(query_files, idf_values,count,doc_tf_idf, doc_len)` calls functions to compute the cosine similarity for each document with 1 query 
* `write_to_file(topicNo, results)` writes the results in a txt file 
* `get_doc_tf_idf(inverted_index,documentsNumbers,documents_max_frequency)` calls functions to create the idf values and the documents tf-idf and stores them in separate files  
* `main()` reads required files and calls the required functions to produce the cosine similarity for 50 queries and rankes them

## How to run the programs 

### Requirements

NLTK is required to run certain functions.

NLTK requires Python versions 2.7, 3.4, 3.5, or 3.6. You can install nltk using pip installer if it is not installed in your Python installation. 

`pip install nltk`

To test the installation: 

Open your Python IDE or the CLI interface (whichever you use normally)
Type `import nltk` and press enter.
If no message of missing nltk is shown then nltk is installed on your computer.

After installation, nltk also provides test datasets to work within Natural Language Processing. One of them (punkt) is used in this project

You can download it by using the following commands in Python:

```
#import the nltk package
import nltk
#call the nltk downloader
nltk.download('punkt')
```
or using this process

![nltk](https://raw.githubusercontent.com/ishanveersg/ishanveer.com/main/nltk.png)

You could also, just type in the python interpreter 
`import nltk`
then 
`nltk.download()`
From the GUI that appears, Click on Models tab and select punkt and click Download.

### Running Step1:Preprocessing and Step2:Indexing

(Note that you do not need to run `Step1:Preprocessing and Step2:Indexing` before running Step3 since we already provided the files need at `/cached` )

The functions for preprocessing step can be found in the `preprocessing.py` file. 

The functions for indexing step can be found in the`indexing.py` file. 

To run the files, you will need to run `main.py` file as follows : 
`python main.py` 

This will create a list of tokens for the preprocessing step stored at `/cached/tokens.json`

It will also create the inverted index for the indexing step stored at `/cached/inverted_index.json`

### Running Step3: Retrieval and Ranking

The functions for the retrieval step can be found in the `retrieval.py` file. 

To calculate the document tf-idf only: 
1. Uncomment line 145
2. Comment lines 147-180 (this calculates the cosine similarity values for 50 queries and ranks them)
3. Run `python retrieval.py` 

(Note that this will take up to 2hours to run)

The results will be stored at `/cached/document-tf-idf`
 
If you decide to calculate the document tf-idf again, after you're done, $don't forget$ to comment line 145 and uncomment lines 147-180 !!!

Else, you can find the calculated document tf-idf at `/cached/document-tf-idf`

To test with 1 query: 
1. Uncomment lines 166-171
2. Comment lines 172-180 (this calculates the cosine similarity values for 50 queries and ranks them)
3. Run `python retrieval.py` 

The results will be stored at `Results.txt`

To test with 50 queries: 

If you tested with 1 query before, 
1. Comment lines 166-171 (this calculates the cosine similarity values for 1 query and ranks them)
2. Uncomment lines 172-180
3. Run `python retrieval.py` 

If you did $not$ test with 1 query before, 
1. Run `python retrieval.py`

The results will be stored at `Results.txt`

## Explanations of algorithms, data structures and optimization

### Step1 : Preprocessing 

In the preprocessing step, we created an algotithms that produces a list of unique tokens based on the content of all the files provided. 
The files are supplied to the program functions that will read them, extract their content and process them. 
The processing first step consist of removing the numbers, punctuations and special characters. 
The result is stemmed (using the nltk porter stemmer ) then tokenized. 
The resulting tokens are then all converted to lower case then processed to remove the duplicates. 
The duplicates are removed using the list and dictionary properties in python.
After, the duplicates are removed, the tokens are then linted to remove the stopwords. 


### Step2 : Indexing

To produce the inverted index, we used the concept of dictionaries where each key is a term and the value is another dictionary with documents numbers as keys and term frequencies as values. 

![inverted index](https://raw.githubusercontent.com/ishanveersg/ishanveer.com/main/inverted_index_example.png)

The function `produce_index(files_names)` uses 3 nested for loops: 
1. To iterate through all the files in the collection
2. To iterate through all the documents in each file 
3. To iterate through each term in each document 

The algorithm is as follow : 

- For each token in the vocabulary, create a key in the inverted index
- For each file supplied, map the documents ( create a dictionary with the document number as key and its content as value)
- For each document mapped, if its contents is not empty, find the most frequent word and count its frequency. 
- for each term in the mapped document content, confirm that it is a key of the inverted index (a token of the vocabulary) and update the count for it in the inverted index
            
### Step3 : Retrieval and Ranking

In this step, we also use a lot of list and dictionaries to store idf values, query and document lengths, document tf idf etc...

All the lists and dictionaries which are used in the calculation of cosine similarities are stored in a json file as soon as they are calculated and retrieved when needed. 
This way, the program does not have to recalculate them since they take a lot of ressources and time to do so. 

### How big is the vocabulary? 

The vocabulary can be found at `\cached\tokens.json`.

The vocabulary is huge. The file size is 1.79MB.
We removed the duplicate words and the stopwords for the collection of documents and also used stemming.

### Sample of 100 tokens 

```
"win", "weekli", "state", "lotteri", "number", "pick", "friday", "lotto", "thi", "play", "wednesday", "megabuck", "bonu", "game", "supplementari", "big", "grand", "lot", "tent", "schedul", "presidenti", "candid", "juli", "inform", "wa", "provid", "sunday", "democrat", "dukaki", "colorado", "jackson", "san", "franicisco", "dalla", "fort", "worth", "texa", "republican", "bush", "washington", "dc", "monday", "boston", "cincinnati", "tuesday", "open", "chicago", "thursday", "indianapoli", "louisvil", "ky", "nashvil", "saturday", "chatanooga", "tenn", "atlanta", "tollfre", "distribut", "agent", "orang", "settlement", "fund", "vietnam", "veteran", "chang", "offici", "previou", "phone", "york", "rest", "countri", "disconnect", "order", "consolid", "aetna", "life", "insur", "claim", "administr", "program", "result", "suit", "manufactur", "herbicid", "elderli", "woman", "toss", "hand", "grenad", "garbag", "dump", "announc", "alreadi" "cant", "attic", "ani", "author", "polic", "block", "area", "learn"
```

### First 10 answers to query 1 and 25 (Using query title only)

When comparing the cosine similarity scores between query 1 and query 25, we can see that the highest score for query 25 is more than 2 times higher than for query 1. 
This means that there are more words in common between query 25 and the collection of documents than between query 1 and the collection of documents.

**First 10 answers to query 1**

```
1 Q0 AP881206-0124 1 0.3163186414144267 testTag
1 Q0 AP881005-0001 2 0.30724343790865416 testTag
1 Q0 AP880825-0054 3 0.2943866308318423 testTag
1 Q0 AP880310-0051 4 0.2735659463733929 testTag
1 Q0 AP880921-0032 5 0.25550549142477547 testTag
1 Q0 AP881002-0014 6 0.25512211678958596 testTag
1 Q0 AP881021-0218 7 0.24109818082779355 testTag
1 Q0 AP881108-0076 8 0.2144976693167953 testTag
1 Q0 AP880815-0061 9 0.21266802594455367 testTag
1 Q0 AP881223-0053 10 0.21000914050093053 testTag
```
**First 10 answers to query 25**

```
25 Q0 AP880917-0094 1 0.6746504342333274 testTag
25 Q0 AP880606-0019 2 0.6622334582655391 testTag
25 Q0 AP880605-0026 3 0.6295751723242442 testTag
25 Q0 AP880812-0017 4 0.5978658177335459 testTag
25 Q0 AP880811-0163 5 0.5940267278534458 testTag
25 Q0 AP881011-0091 6 0.5798469052978871 testTag
25 Q0 AP880427-0150 7 0.5759855703295357 testTag
25 Q0 AP880427-0240 8 0.5579615688809495 testTag
25 Q0 AP881016-0013 9 0.5122612525845472 testTag
25 Q0 AP880916-0009 10 0.5086867404307712 testTag

```

### First 10 answers to query 1 and 25 (Using query title and description )

When comparing the results for the first 10 answers for query 1 and query 25 using the query title and description, we can see that query 25's cosine similarities are higher than query 1's cosine similarities. However, they do not differ by much as compared to the comparison for the query title only. This might be because using the query title and description gives a larger vocabulary, thus there might be more similarities between query 1 and query 25 than when using the title only. 

**First 10 answers to query 1**

```
1 Q0 AP881206-0124 1 0.4608100836429267 testTag
1 Q0 AP881005-0001 2 0.34827045965178083 testTag
1 Q0 AP880815-0061 3 0.3242597036782592 testTag
1 Q0 AP880825-0054 4 0.312526378118919 testTag
1 Q0 AP881021-0218 5 0.29719831537614294 testTag
1 Q0 AP881002-0014 6 0.2887703486506893 testTag
1 Q0 AP881223-0053 7 0.2755659962089984 testTag
1 Q0 AP880726-0173 8 0.2752180220965938 testTag
1 Q0 AP880814-0089 9 0.2605719128211265 testTag
1 Q0 AP881108-0076 10 0.25626497974619333 testTag

```
**First 10 answers to query 25**

```
25 Q0 AP880917-0094 1 0.583079872916989 testTag
25 Q0 AP880606-0019 2 0.5732649226007884 testTag
25 Q0 AP880605-0026 3 0.5445313241264977 testTag
25 Q0 AP880812-0017 4 0.5191886766426703 testTag
25 Q0 AP880811-0163 5 0.5182857035540346 testTag
25 Q0 AP881011-0091 6 0.504376630227362 testTag
25 Q0 AP880427-0150 7 0.4986272940129178 testTag
25 Q0 AP880427-0240 8 0.4821670725373924 testTag
25 Q0 AP881016-0013 9 0.447260309406706 testTag
25 Q0 AP880916-0009 10 0.43738459689439657 testTag

```


### Comparing performance for query title VS query title and description 

Using the query title and description gives a better performance than using the title only. 

Using trec_eval, we can see that for the number of relevant documents using the title only is 1984 as compared to 2096 number of relevant documents using title and description. 

Also, the mean average precision is better by approximately 0.02 when using title and description. 

This might be because using the query title and description gives a larger vocabulariy, thus more similarities between the queries and the documents. 

## Mean Average Precision (MAP)

The Mean Average Precision when using the query title only is 0.2159

![Results using query title only](https://raw.githubusercontent.com/ishanveersg/ishanveer.com/main/results_title.png)

The Mean Average Precision when using the query title and description is 0.2368

![Results using query title and description](https://raw.githubusercontent.com/ishanveersg/ishanveer.com/main/results_title_desc.png)

We can see that the MAP is better when using the query title and description than just using the title. This may be because the vocabulary is bigger using the description as well and thus there are more similarities between the collection of documents and the query. 

## References 

 Remove Duplicates: https://www.w3schools.com/python/python_howto_remove_duplicates.asp

 chatGPT : https://chat.openai.com/chat 

 Week2 slides : https://www.site.uottawa.ca/~diana/csi4107/L3.pdf 

 Stemming : https://www.datacamp.com/tutorial/stemming-lemmatization-python

 cosine similarity : https://www.sciencedirect.com/topics/computer-science/cosine-similarity#:~:text=The%20measure%20computes%20the%20cosine,greater%20the%20match%20between%20vectors. 
