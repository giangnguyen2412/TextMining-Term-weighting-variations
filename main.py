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
import itertools
import timeit

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
dir = ('20news-bydate/*/*')
WEIGHT_CSV = 'self_weight.csv'
WEIGHT_CSV_SORT = 'self_weight_sort.csv'
EXPECTED_TERM_NUM = 50

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
    occurence_mat_ret = np.zeros((docs_num, vocab_size))
    df_mat = np.zeros((docs_num, vocab_size))
    occurence_cnt_mat = np.zeros((docs_num, vocab_size))
    for text in docs:   # Iterate through all documents
        prep_text_l = list(preprocess(text))
		
        prep_text_l_clone = prep_text_l.copy()
        prep_text_s = set(prep_text_l_clone)

        row = docs.index(text)
        col = [vocabulary.index(token) for token in prep_text_s]
        df_mat[row, col] += 1

        c = Counter(prep_text_l)
        for word in prep_text_l:
            term_freq = (c[word])
            occurence_cnt_mat[row, vocabulary.index(word)] += term_freq

    occurence_mat_ret = occurence_cnt_mat
    df_mat = np.sum(df_mat, axis = 0)
    tf_mat = np.sum(occurence_cnt_mat, axis = 0)

    idf_val = np.log(docs_num/df_mat)
    tf_val = tf_mat/vocab_size

    # Return this value to get tf-idf weights
    tf_idf_w = np.multiply(idf_val,tf_val)

    return idf_val, occurence_mat_ret

def compute_log_entropy(docs, vocabulary_list, occur_matrix):
    # An array to store occurence frequency
    frequency_mat = np.zeros((docs_num, vocab_size))
    cVocabulary = Counter(vocabulary_list)
    for text in docs:   # Iterate through all cocuments
        cText = Counter(text)
        prep_text_l = list(preprocess(text))
        row = docs.index(text)

        for word in prep_text_l:
            col = vocabulary.index(word)
            frequency_mat[row, col] += cText[word]/cVocabulary[word]

    local_w = occur_matrix + 1;
    local_w = np.log(local_w)
    local_w = np.mean(local_w, axis = 0)
	
    frequency_matrix_log = np.log(frequency_mat + 1) # Add 1 to avoid inf numbers
    mul = np.multiply(frequency_matrix_log,frequency_mat);
    mul_sum = np.sum(mul, axis = 0)
    global_w = 1 +  mul_sum/(np.log(docs_num+1))
    final_w = np.multiply(local_w,global_w);
	
    log_entropy_w = final_w
    return log_entropy_w

def compute_tdv(docs):
    occurence_freq_mat = np.zeros((docs_num, vocab_size))
    tdv_mat = np.zeros((docs_num, vocab_size))
    for text in docs:   # Iterate through all documents
        prep_text_l = list(preprocess(text))
		
        prep_text_l_clone = prep_text_l.copy()
        prep_text_s = set(prep_text_l_clone)

        row = docs.index(text)
        col = [vocabulary.index(token) for token in prep_text_s]

        c = Counter(prep_text_l)
        for word in prep_text_l:
            term_freq = (c[word])
            occurence_freq_mat[row, vocabulary.index(word)] += term_freq

    #print(np.shape(occurence_freq_mat))
    #input("Press Enter to continue...")

    # ---------------------------------------------------------------
    # The above part of code can be replaced simply by:
    '''
    words_tdv = np.zeros((docs_num, vocab_size))
    tf_vectorizer = CountVectorizer(tokenizer=preprocess, stop_words=stopWords)

    docs_terms = tf_vectorizer.fit_transform(docs)
    occurence_num_matrix_test = docs_terms.todense()	# To get the matrix of token counts
    print(np.shape(occurence_num_matrix_test))
    compare_matrix = (occurence_num_matrix_test == occurence_freq_mat)    # For testing validity, expect to be True
    print(compare_matrix)
    input("Press Enter to continue...")
    '''
    centroid = np.mean(occurence_freq_mat, axis = 0)
    for i in range(docs_num):
        doc = occurence_freq_mat[i,:]
        avg_sim = np.linalg.norm(centroid - doc)
        for j in range(vocab_size):
            if (doc[j] != 0):
                doc_t = doc
                doc_t[j] = 0;
                avg_sim_m = np.linalg.norm(centroid - doc_t)
                tdv_mat[i, j] = avg_sim - avg_sim_m

    tdv_w = np.mean(tdv_mat, axis = 0)

    return tdv_w
	

def find_tuples(tup, num):
    return [i for i in itertools.permutations(tup, num)]
	
def calculate_avg_sim(occurence_mat, vocab_tup):
	''' 
	Bcz we want to get pairs so num is 2
	The pairs are repeated, for example (i,j) and (j,i),
	thn we need to divide by 2 for calculating Similarity
	'''
	pair_tup = find_tuples(vocab_tup, 2)	
	avg_sim = 0
	for i in range(len(pair_tuple)):
		# Get each document pair first
		first_doc_num = (pair_tup[i])[0]
		secon_doc_num = (pair_tup[i])[1]
		
		first_doc = occurence_mat[first_doc_num, :]
		secon_doc = occurence_mat[secon_doc_num, :]
		avg_sim += np.linalg.norm(first_doc - secon_doc)
		
	avg_sim = (avg_sim/(docs_num*(docs_num-1)))/2
	return avg_sim
	
	
def compute_tdv_v2(docs):
	occurence_mat = np.zeros((docs_num, vocab_size))
	# Temporary value of occurence_mat
	occurence_mat_t = np.zeros((docs_num, vocab_size))
	
	avg_sim_mat = np.zeros((1, vocab_size))
	words_sn = np.zeros((docs_num, vocab_size))
	tf_vectorizer = CountVectorizer(tokenizer=preprocess, stop_words=stopWords)
	
	docs_terms = tf_vectorizer.fit_transform(docs)
	occurence_mat = docs_terms.todense()	# To get the matrix of token counts
	
	vocab_tup = ()
	for i in range(docs_num):
		vocab_tup += (i,)
		
	avg_sim = calculate_avg_sim(occurence_mat, vocab_tup)
	
	zeros_col = np.zeros(docs_num)
	avg_sim_m = 0
	for i in range(vocab_size):
		occurence_mat_t = np.array(occurence_mat)
		occurence_mat_t[:, i] = zeros_col 	# Remove occurence of term i-th from collection
		avg_sim_m = calculate_avg_sim(occurence_mat_t, vocab_tup)
		avg_sim_mat[0,i] = avg_sim_m
		
	avg_sim_mat = avg_sim_mat - avg_sim
	return avg_sim_mat;
		
	'''
	#print(np.count_nonzero(occurence_mat_t[:,i]))
	#print(np.count_nonzero(occurence_mat_t[:,i+1]))
	#input("Press Enter to continue...")
	print(occurence_mat)
	print(np.count_nonzero(occurence_mat))
	input("Press Enter to continue...")
	print(np.count_nonzero(occurence_mat_t))
	print(avg_sim)
	input("Press Enter to continue...")
	'''

	
def compute_signal_noise(docs):
	## FOR THIS ADDITIONAL PART, WE WILL USE CountVectorizer to get occurence_freq_mat ##
    occurence_mat = np.zeros((docs_num, vocab_size))
    occurence_freq_mat = np.zeros((1, vocab_size))
    tf_vectorizer = CountVectorizer(tokenizer=preprocess, stop_words=stopWords)

    docs_terms = tf_vectorizer.fit_transform(docs)
    occurence_mat = docs_terms.todense()	# To get the matrix of token counts
    occurence_mat += 1	# Add 1 to avoid log error

    occurence_freq_mat = np.sum(occurence_mat, axis = 0)
    occurence_freq_mat_clone = np.tile(occurence_freq_mat, (docs_num,1));
	
    p_im = np.divide(occurence_mat, occurence_freq_mat_clone) # occurence_mat/clone_occurence_num_matrix
    log_p_im = np.log(1/p_im)
	
    multiply = np.multiply(p_im, log_p_im)
    Nm = np.sum(multiply, axis = 0)
    fm = np.log(occurence_freq_mat) - Nm
	
    return np.squeeze(np.asarray(fm))


def get_files(dir):
    global docs
    global docs_num
    files = glob.glob(dir, recursive=True)
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

def sort_and_export(idf_weights, tdv_weights, entropy_weights, sn_weights, filename):
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
	
    sn_term_list = pd.Series(sn_weights)
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
    start = timeit.default_timer()
    global docs
    global docs_name
    global vocabulary
    global vocab_size

    docs,docs_name = get_files(dir)
	
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
    if_idf_w, occur_matrix = compute_tf_idf(docs)

    vocabulary_l = [item for sublist in vocab_l for item in sublist]
	
    log_entropy_w = compute_log_entropy(docs, vocabulary_l, occur_matrix)
	
	# Can get occur_matrix and vocabulary_l from above to speed up
    tdv_w = compute_tdv(docs)
    #tdv_weights_v2 = compute_tdv_v2(docs)
    signal_noise_w = compute_signal_noise(docs)

    sort_and_export(if_idf_w, tdv_w, log_entropy_w, signal_noise_w, WEIGHT_CSV_SORT)
    stop = timeit.default_timer()
    print('Eslaped Time: ', stop - start)  
    # Just debugging
    #c = Counter(flat_list)
    #print(len(c))


if __name__ == '__main__':
    main()
