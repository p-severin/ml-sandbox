import re
from collections import Counter

import numpy
import tqdm

import nltk
from nltk.corpus import gutenberg
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer

from embeddings import initialize_embeddings

embedding_size = 200
window_size = 5
no_negative_samples = 15

tokens = None
texts = nltk.corpus.gutenberg.fileids()

def sliding_window(words_from_text: list, window_size: int):
    n_words = len(words_from_text)
    return [words_from_text[i:(n_words - window_size)] for i in range(n_words)]

def generate_tokens(titles):
    corpus = []
    for title in titles:
        novel: str = gutenberg.raw(title)
        novel = novel.strip()
        novel = novel.lower()
        novel = re.sub('\W+', ' ', novel)
        words = novel.split(' ')
        corpus.extend(words)
    return corpus

def get_unique_words(corpus: list):
    return set(corpus)

def remove_useless_words(corpus: list, vocabulary):
    for word in tqdm.tqdm(corpus):
        if word not in vocabulary:
            corpus.remove(word)
    return corpus


def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y = [], []

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])

    X = numpy.array(X)
    # X = numpy.expand_dims(X, axis=0)
    Y = numpy.array(Y)
    # Y = numpy.expand_dims(Y, axis=0)

    return X, Y

def get_mappings(tokens):
    word_to_id = dict()
    id_to_word = dict()
    unique_words = get_unique_words(tokens)
    for i, word in enumerate(unique_words):
        word_to_id[word] = i
        id_to_word[i] = word

    return word_to_id, id_to_word


if __name__ == '__main__':
    tokens = generate_tokens(texts[:2])
    unique_words = get_unique_words(tokens)
    nb_tokens = len(unique_words)
    word_to_id, id_to_word = get_mappings(tokens)

    X, y = generate_training_data(tokens, word_to_id, window_size)
    y_one_hot = numpy.zeros((len(X), nb_tokens))
    y_one_hot[numpy.arange(len(X)), y] = 1

    embeddings = initialize_embeddings(nb_tokens, embedding_size)
    print(embeddings)

    # counter = Counter(tokens)
    # print(X)
    # vectorizer = CountVectorizer(min_df=10, stop_words=stop_words.ENGLISH_STOP_WORDS)
    # transformed = vectorizer.fit_transform(tokens)
    # vocabulary = vectorizer.vocabulary_.keys()
    # print(len(tokens))



