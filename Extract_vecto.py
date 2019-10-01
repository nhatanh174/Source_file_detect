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
import TF_IDF

# vector of line
def CombineVector(line_sent,corpus):
    vector_line = np.zeros(shape=(1,300))
    count=0
    for word_sent in  line_sent:
        vector_line += w2v_model.wv[word_sent]*TF_IDF.computeTf_Idf(corpus,word_sent)
        count+=1
    vector_line=vector_line/count
    return vector_line
# vecto of sourfile
def Vector(corpus):
    number_line=[]
    for sourcefile in corpus:
        number_line.append(len(sourcefile))
    line_max= max(number_line)
    matrix=[]
    for sourcefile in corpus:
        vecto=np.zeros(shape=(line_max,300))
        for i in range(sourcefile):
            vecto[i]= CombineVector(sourcefile[i],corpus)
        matrix.append(vecto)
    return matrix

