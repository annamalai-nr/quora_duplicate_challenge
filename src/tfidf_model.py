
import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from classify import *

from utils import load_dataset

def get_vocab_from_all_questions(question1,question2):
    unique_questions = set(question1 + question2)
    vectorizer.fit(unique_questions)
    return vectorizer.get_feature_names()

question_pairs, labels = load_dataset()
print 'loaded {} question paris'.format(len(question_pairs))

vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,3),stop_words='english')
question1,question2 = zip(*question_pairs)
vocab = get_vocab_from_all_questions(question1,question2)
print 'extracted {} features from question pairs'.format(len(vocab))

vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,3),stop_words='english',vocabulary=vocab)
x_1 = vectorizer.fit_transform(question1)
x_2 = vectorizer.fit_transform(question2)
x = hstack([x_1,x_2])
svm_fit(x,labels)



