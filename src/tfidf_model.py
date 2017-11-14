import numpy as np
import sys
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from classify import *

from utils import load_dataset
from collections import Counter
from sklearn.decomposition import TruncatedSVD


def get_vocab_from_all_questions(question1,question2):
    unique_questions = set(question1 + question2)
    vectorizer.fit(unique_questions)
    return vectorizer.get_feature_names()

question_pairs, labels = load_dataset(load_n=None)
print 'loaded {} question paris'.format(len(question_pairs))
print 'pos and neg sample counts', Counter(labels)

vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,3),stop_words='english')
question1,question2 = zip(*question_pairs)

vocab = get_vocab_from_all_questions(question1,question2)
print 'extracted {} features from question pairs'.format(len(vocab))

vectorizer = TfidfVectorizer(lowercase=True,ngram_range=(1,3),
                             stop_words='english',vocabulary=vocab)
x_1 = vectorizer.fit_transform(question1)
x_2 = vectorizer.fit_transform(question2)
# svd = TruncatedSVD(n_components=1000)
# x_1 = svd.fit_transform(x_1)
# x_2 = svd.fit_transform(x_2)
try:
    x = hstack([x_1,x_2])
except:
    x = np.hstack([x_1,x_2])

# for t in [0.6,0.7,0.8,0.9]:
#     threshold_sim_eval(x_1,x_2,labels,threshold=0.9)
svm_fit(x,labels)
# xgboost_fit(x,labels)
# mlp_fit(x,labels)



