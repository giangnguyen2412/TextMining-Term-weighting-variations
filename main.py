import glob
import os
import nltk
import numpy as np
import pandas as pd

from csv import DictWriter
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from collections import Counter

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

word_list = set(nltk_words.words())

docs_num = 0
docs = []
docs_name = []
vocabulary = []
vocab_size = 0
directory = ('20news-bydate/*/*') # We should you tuple for string

def preprocess(text):
    # I actually think doing both stemming and lemmatizing is a redundance
    stemmed = []
    lemmatized = []
    wordsFiltered = []

    # Normalization
    text = text.lower()

    # Tokenization
    token_text = nltk.word_tokenize(text)

    # Stemming
    for token in token_text:
        stemmed.append(porter_stemmer.stem(token))

    # Lemmatization
    for token in stemmed:
        lemmatized.append(wordnet_lemmatizer.lemmatize(token))

    wordsFiltered = filter(lambda word_to_filter: word_to_filter not in stopWords and word_to_filter.isalpha() and word_to_filter in word_list, lemmatized)

    vocab_entries = wordsFiltered
    return vocab_entries
'''
    # Filtering using stopwords
    for token in lemmatized:
        if token not in stopWords:
            wordsFiltered.append(token)
            '''

def compute_tf_idf(docs):
    term_occurence_matrix_ret = np.zeros((docs_num, vocab_size))
    occurence_matrix = np.zeros((docs_num, vocab_size))
    occurence_num_matrix = np.zeros((docs_num, vocab_size))
    for text in docs:   # Iterate through all cocuments
        # set() and list() seem to be erase value of input imme, so I need to perform preprocess 2 times => Need to improved
        preprocessed_text = preprocess(text)
        preprocessed_text_set = set(preprocessed_text)

        preprocessed_text_l = preprocess(text)
        preprocessed_text_list = list(preprocessed_text_l)

        row = docs.index(text)
        col = [vocabulary.index(token) for token in preprocessed_text_set]
        occurence_matrix[row, col] += 1

        c = Counter(preprocessed_text_list)
        for word in preprocessed_text_list:
            term_freq = (c[word])
            occurence_num_matrix[row, vocabulary.index(word)] += term_freq

    term_occurence_matrix_ret = occurence_num_matrix
    term_occurence_matrix = np.sum(occurence_matrix, axis = 0)
    term_occurence_num_matrix = np.sum(occurence_num_matrix, axis = 0)

    print(term_occurence_matrix)
    print(term_occurence_num_matrix)

    print(len(term_occurence_matrix))
    print(len(term_occurence_num_matrix))

    idf_value = np.log(docs_num/term_occurence_matrix)
    tf_value = term_occurence_num_matrix/vocab_size

    # Return this value to get tf-idf weights
    tf_idf_weight = np.multiply(idf_value,tf_value)

    print(tf_idf_weight)
    return idf_value, term_occurence_matrix_ret

def compute_log_entropy(docs, vocabulary_list, occur_matrix):
    frequency_matrix = np.zeros((docs_num, vocab_size))
    cVocabulary = Counter(vocabulary_list)
    for text in docs:   # Iterate through all cocuments
        cText = Counter(text)
        preprocessed_text = list(preprocess(text))
        row = docs.index(text)

        for word in preprocessed_text:
            col = vocabulary.index(word)
            frequency_matrix[row, col] += cText[word]/cVocabulary[word]

    local_w = occur_matrix + 1;
    local_weight_t = np.log(local_w)
    local_weight = np.mean(local_weight_t, axis = 0)
    frequency_matrix_log = np.log(frequency_matrix + 1) # Add 1 to avoid inf numbers
    mul = np.multiply(frequency_matrix_log,frequency_matrix);
    mul_sum = np.sum(mul, axis = 0)
    global_weight = 1 +  mul_sum/(np.log(docs_num+1))
    final_weight = np.multiply(local_weight,global_weight);
    log_entropy_weight = final_weight
    return log_entropy_weight

def compute_tdv(docs):
    # -------------------------------------------------------------
    #occurence_matrix = np.zeros((docs_num, vocab_size))
    occurence_num_matrix = np.zeros((docs_num, vocab_size))
    tdv_matrix = np.zeros((docs_num, vocab_size))
    for text in docs:   # Iterate through all cocuments
        # set() and list() seem to be erase value of input imme, so I need to perform preprocess 2 times => Need to improved
        preprocessed_text = preprocess(text)    # preprocessed_text To get tokenized text in "unique" (set) format
        preprocessed_text_set = set(preprocessed_text)

        preprocessed_text_l = preprocess(text)  # preprocessed_text_l To get tokenized text in format with removing duplicate words
        preprocessed_text_list = list(preprocessed_text_l)

        row = docs.index(text)
        col = [vocabulary.index(token) for token in preprocessed_text_set]

        c = Counter(preprocessed_text_list)
        for word in preprocessed_text_list:
            term_freq = (c[word])
            occurence_num_matrix[row, vocabulary.index(word)] += term_freq

    #print(np.shape(occurence_num_matrix))
    #input("Press Enter to continue...")

    # ---------------------------------------------------------------
    # The above part of code can be replaced simply by:
    '''
    words_tdv = np.zeros((docs_num, vocab_size))
    tf_vectorizer = CountVectorizer(tokenizer=preprocess, stop_words=stopWords)

    docs_terms = tf_vectorizer.fit_transform(docs)
    occurence_num_matrix_test = docs_terms.todense()	# To get the matrix of token counts
    print(np.shape(occurence_num_matrix_test))
    compare_matrix = (occurence_num_matrix_test == occurence_num_matrix)    # For testing validity, expect to be True
    print(compare_matrix)
    input("Press Enter to continue...")
    '''
    centroid = np.mean(occurence_num_matrix, axis = 0)
    for i in range(docs_num):
        doc = occurence_num_matrix[i,:]
        #print(np.shape(doc))
        #input("Press Enter to continue...")
        avg_sim = np.linalg.norm(centroid - doc)
        for j in range(vocab_size):
            #print(type(doc[j]))
            #input("Press Enter to continue...")
            if (doc[j] != 0):
                doc_t = doc
                doc_t[j] = 0;
                avg_sim_m = np.linalg.norm(centroid - doc_t)
                tdv_matrix[i, j] = avg_sim - avg_sim_m

    tdv_weight = np.mean(tdv_matrix, axis = 0)

    return tdv_weight
	
def compute_signal_noise(docs):
	## -------------------- FOR THIS ADDITIONAL PART, WE WILL USE CountVectorizer to get occurence_num_matrix ##
    occurence_matrix = np.zeros((docs_num, vocab_size))
    occurence_num_matrix = np.zeros((1, vocab_size))
    words_sn = np.zeros((docs_num, vocab_size))
    tf_vectorizer = CountVectorizer(tokenizer=preprocess, stop_words=stopWords)

    docs_terms = tf_vectorizer.fit_transform(docs)
    occurence_matrix = docs_terms.todense()	# To get the matrix of token counts
    occurence_matrix += 1	# Add 1 to avoid log error

    occurence_num_matrix = np.sum(occurence_matrix, axis = 0)
    clone_occurence_num_matrix = np.tile(occurence_num_matrix, (docs_num,1));
	
    p_im = np.divide(occurence_matrix, clone_occurence_num_matrix) # occurence_matrix/clone_occurence_num_matrix
    log_p_im = np.log(1/p_im)
	
    multiply = np.multiply(p_im, log_p_im)
    Nm = np.sum(multiply, axis = 0)
    fm = np.log(occurence_num_matrix) - Nm
    print(np.shape(fm))
    return np.squeeze(np.asarray(fm))


def get_files_from_dir(directory):
    global docs
    global docs_num
    files = glob.glob(directory, recursive=True)
    for file in files:
        with open(file, mode='r') as f:
             docs.append(f.read())
             docs_name.append(file)
    docs_num = len(files)
    return docs, docs_name
'''
    print(docs[1])
    print(files[1])
    print(len(files))
    '''

WEIGHT_CSV = 'self_weight.csv'
WEIGHT_CSV_SORT = 'self_weight_sort.csv'
EXPECTED_TERM_NUM = 50

def export_to_file_sort(idf_weights, tdv_weights, entropy_weights, signal_noise_weights, filename):
    entropy_arr = np.squeeze(np.asarray(entropy_weights))
    idf_term_list = pd.Series(idf_weights)
    idf_w_des = idf_term_list.sort_values(ascending=False)
    idf_w_des_idx = idf_w_des.index

    tdv_term_list = pd.Series(tdv_weights)
    tdv_w_des = tdv_term_list.sort_values(ascending=False)
    tdv_w_des_idx = tdv_w_des.index

    entropy_term_list = pd.Series(entropy_arr)
    entropy_w_asc = entropy_term_list.sort_values(ascending=True)
    entropy_w_asc_idx = entropy_w_asc.index
	
    sn_term_list = pd.Series(signal_noise_weights)
    sn_w_des = sn_term_list.sort_values(ascending=False)
    sn_w_des_idx = sn_w_des.index
	
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['idf_term_sort', 'idf_weight_sort','tdv_term_sort', 'tdv_weight_sort','entropy_term_sort', 'entropy_weight_sort', 'S/N_term_sort', 'S/N_weight_sort']
        writer = DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(vocab_size):
            writer.writerow({
                'idf_term_sort':vocabulary[idf_w_des_idx[i]],
                'idf_weight_sort': idf_w_des.values[i],
                'tdv_term_sort':vocabulary[tdv_w_des_idx[i]],
                'tdv_weight_sort': tdv_w_des.values[i],
                'entropy_term_sort':vocabulary[entropy_w_asc_idx[i]],
                'entropy_weight_sort': entropy_w_asc.values[i],
                'S/N_term_sort':vocabulary[sn_w_des_idx[i]],
                'S/N_weight_sort': sn_w_des.values[i],				
            })
    print('Weight result has been saved to', filename)

def main():
    global docs
    global docs_name
    global vocabulary
    global vocab_size

    docs,docs_name = get_files_from_dir(directory)
    '''
    print(docs)
    print(docs_name)
    docs_name_arr = np.asarray(docs_name)
    print(docs_name_arr.shape)
    '''
    # We use set to ensure unique of a term in vocab
    vocab = set()
    vocab_l = list()

    for text in docs:
        tokens = preprocess(text)
        tokens_l = preprocess(text)
        vocab.update(tokens)
        vocab_l.append(list(tokens_l))

    vocabulary = sorted(vocab, reverse = False)
    vocab_size = len(vocabulary)

    occur_matrix = np.zeros((docs_num, vocab_size))
    if_idf_weight, occur_matrix = compute_tf_idf(docs)

    vocabulary_list = [item for sublist in vocab_l for item in sublist]
    log_entropy_weight = compute_log_entropy(docs, vocabulary_list, occur_matrix)
    tdv_weights = compute_tdv(docs)
    signal_noise_weights = compute_signal_noise(docs)

    export_to_file_sort(if_idf_weight, tdv_weights, log_entropy_weight, signal_noise_weights, WEIGHT_CSV_SORT)

    # Just debugging
    #c = Counter(flat_list)
    #print(len(c))


if __name__ == '__main__':
    main()
