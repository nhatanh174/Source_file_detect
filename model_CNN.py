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
def feature_detect (matrix):
    vectors=[]
    for source in matrix:
        input= np.expand_dims(source,axis=0)
        vector_feature= np.zeros((3,100))
        num=0
        for i in range(2,5):
            model = Sequential()
            model.add(Conv1D(activation='relu', filters=100, kernel_size=i, input_shape=source.shape))
            model.add(MaxPooling1D(pool_size=source.shape[0] - i + 1))
            model.compile(loss=CategoricalCrossentropy, metrics=['accuracy'],optimizer=Adam)
            vector= np.reshape(model.predict(input),(1,100))
            vector_feature[num]=vector
            num+=1
        vectors.append(vector_feature.flatten())
    return vectors