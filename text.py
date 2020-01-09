import gensim

import nltk
# The 'english' stemmer is better than the original 'porter' stemmer.
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.corpus import wordnet

import numpy as np


stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)


def lemmatize_stemming(text):
    '''
    Stemmer causes too many invalid words.
    '''
    #return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
    return lemmatizer.lemmatize(text, pos=get_wordnet_pos(text))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    # simple_preprocess(): lowercases, tokenizes, de-accents (optional)
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))

    return result
