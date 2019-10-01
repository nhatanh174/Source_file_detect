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
import ntpath

# function read folder
def openFolder(path, files, agr):
    files.extend(glob.glob(os.path.join(path, agr)))
    for file in os.listdir(path):
        fullpath = os.path.join(path, file)
        if os.path.isdir(fullpath) and not os.path.islink(fullpath):
            openFolder(fullpath,files,agr)
# get month and year of
def getMonth(file):
    t = os.path.getmtime(file)
    a = datetime.datetime.fromtimestamp(t)
    return (a.date().month,a.date().year)
# get name of file
def getName(files):
    for i in range(0,len(files)):
        files[i]=ntpath.basename(files[i])
    return files