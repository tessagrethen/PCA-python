# Name: Tessa Pham
# File: PCA.py
# Implements the PCA method to analyze and compare texts.

# process the Brown corpus
# calculate the frequencies of the function words

import nltk
from nltk.corpus import *

# import pandas as pd
import numpy as np
from numpy import array, mean, cov
from numpy.linalg import eig, svd

# build a tagger based on the Brown tagged corpus
def CascadingTaggers():
    btw = brown.tagged_words()
    bts = brown.tagged_sents()

    # find the most frequent tag in the corpus
    tags = [tag for (word, tag) in btw]
    tagDist = nltk.FreqDist(tags)
    maxTag = tagDist.max()

    # build a Defaut Tagger with the most frequent tag
    dTagger = nltk.DefaultTagger(maxTag)

    # build a Unigram Tagger trained on bts and backed up by Default
    udTagger = nltk.UnigramTagger(bts, backoff = dTagger)

    # build a Bigram Tagger trained on bts and backed up by Unigram
    budTagger = nltk.BigramTagger(bts, backoff = udTagger)

    # build a Trigram Tagger trained on bts and backed up by Bigram
    tbudTagger = nltk.TrigramTagger(bts, backoff = budTagger)

    return tbudTagger

def getFunctionWords(text):
    # lists of function POS
    functionPOS = ['ABL', 'ABN', 'ABX', 'AP', 'AT', 'BE', 'BED', 'BEDZ', 'BEG', 'BEM', 'BEN', 'BER', 'BEZ', 'CC', 'CD', 'CS', 'DO', 'DOD', 'DOZ', 'DT', 'DTI', 'DTS', 'DTX', 'EX', 'HV', 'HVD', 'HVG', 'HVN', 'HVZ', 'IN', 'MD', 'NR', 'NRS', 'OD', 'PN', 'PN$', 'PP$', 'PP$$', 'PPL', 'PPLS', 'PPO', 'PPS', 'PPSS', 'QL', 'QLP', 'RP', 'TO', 'WDT', 'WPO', 'WPS', 'WQL', 'WRB']
    functionWordsFreq = nltk.FreqDist()

    # process the text with Cascading Taggers
    tagger = CascadingTaggers()
    words = [w.lower() for w in nltk.word_tokenize(text)]
    taggedWords = tagger.tag(words)

    for (word, tag) in taggedWords:
        if tag in functionPOS:
            functionWordsFreq[word] += 1
    
    return [word for word, count in functionWordsFreq.most_common(10)]

# varList = getFrequentFunctionWords(biggerText)
def getVector(text, varList):
    words = nltk.word_tokenize(text)
    vector = []

    for var in varList:
        count = 0
        for w in words:
            if w == var:
                count += 1
        vector.append(count)
    
    return vector

"""
# find center of data
def findCenter(vectorList):
    center = []
    numVars = len(vectorList[0])

    for i in range(0, numVars):
        value = 0
        for vector in vectorList:
            value += vector[i]
        value /= len(vectorList)
        center.append(value)
    
    return center

def recenterData(vectorList, center):
    # recentered vector list
    recentered = []

    for vector in vectorList:
        vector = [v - c for v, c in zip(vector, center)]
        recentered.append(vector)
    
    return recentered
"""

# find a line that goes through center and best describes all other vectors

def main():
    bigText = inaugural.raw('2009-Obama.txt')
    varList = getFunctionWords(bigText)
    samples = [para.lower() for para in bigText.split('\n\n')[1:6]]
    vectorList = [getVector(text, varList) for text in samples]
    
    # create a matrix, each vector is a row, 10 columns for 10 variables
    A = array(vectorList)
    # find the empirical mean vector
    M = mean(A, axis = 0)
    # recenter the matrix
    C = A - M

    
# if __name__ == "__main__":
#    main()


    
