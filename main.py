import glob
import os
import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words as nltk_words

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
directory = ('20news-bydate/*/*')

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

    tf_idf_weight = idf_value*tf_value
    tf_idf_weight = np.mean(tf_idf_weight, axis=0)

    print(tf_idf_weight)
    return tf_idf_weight, term_occurence_matrix_ret

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
    local_weight = np.mean(local_weight_t, axis=0)
    frequency_matrix_log = np.log(frequency_matrix + 1) # Add 1 to avoid inf numbers
    mul = np.multiply(frequency_matrix_log,frequency_matrix);
    mul_sum = np.sum(mul, axis=0)
    global_weight = 1 +  mul_sum/(np.log(docs_num+1))
    final_weight = np.multiply(local_weight,global_weight);
    log_entropy_weight = final_weight
    return log_entropy_weight


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

    # Just debugging
    #c = Counter(flat_list)
    #print(len(c))


if __name__ == '__main__':
    main()
