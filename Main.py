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
import File as file
import Data_processing as pre
import Extract_vecto as vecto
import model_CNN
path= r"C:\Users\Dell\PycharmProjects\BugDetect\SourceFile"

files=[]        # list name file.java
file.openFolder(path,files,'*.java')
# list sourfile
corpus=[]
for file in files:
    corpus.append(pre.pre_processing(file))
print(corpus)
# list matrix of sourfiles
matrix= vecto.Vector(corpus)
#list vecto feature
vecto_detec=model_CNN.feature_detect(matrix)
















