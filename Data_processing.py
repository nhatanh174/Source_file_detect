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
#pre_processing
def pre_processing(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    f.close()
    sentences = []
    for newline in lines:
        x = []
        line = re.sub(r'\W', ' ', newline)
        line = re.sub(r'\d', '', line)
        line = re.sub('_', ' ', line)
        words = word_tokenize(line)
        if len(line) <= 1:
            continue
        for word in words:
            if word.islower() == True:
                x.append(word)
            elif word.isupper() == True:
                x.append(word.lower())
            else:
                for i in range(1, len(word)):
                    if word[i].isupper() == True:
                        word= word[:i] + ' ' + word[i:]
                        continue
                for ptu in word_tokenize(word):
                    x.append(ptu.lower())
        if len(x) != 0:
            sentences.append(x)
    return  sentences
