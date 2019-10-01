from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import multiprocessing
from gensim.models import Word2Vec,KeyedVectors
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Flatten,Input,Embedding
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import os
import glob
import pandas as pd
import datetime
# compute TF-IdF
def computeTf_Idf(corpus,pharse):
    array = []
    for i in range(0, len(corpus)):
        flatten = [j for sub in corpus[i] for j in sub]
        array.append(' '.join(flatten))
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(array)
    first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]
    df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                      columns=["tfidf"])
    return df['tfidf'][pharse]