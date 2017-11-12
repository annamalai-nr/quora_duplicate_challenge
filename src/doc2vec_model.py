import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import load_dataset
import numpy as np
import psutil,json,random
from pprint import pprint
from classify import *


def load_pretrained_word_vectors (word_vector_fname='../embeddings/glove.6B.300d.txt', vocab=None):
    word_vectors = {}
    with open(word_vector_fname,'r') as fh:
        for line in fh.xreadlines():
            values = line.strip().split()
            word = values[0]
            if word in vocab:
                coefs = np.asarray(values[1:], dtype='float32')
                word_vectors[word] = coefs
    return word_vectors

def init_word_vectors_in_model (model,index2word,word_vectors):
    for index,word in enumerate(index2word):
        try:
            vector = np.array(word_vectors[word])
            model.syn1neg[index] = vector
            # print 'loaded vector for ', word
        except:
            continue


question_pairs, labels = load_dataset(load_n=None)
print 'loaded {} question paris'.format(len(question_pairs))


question1,question2 = zip(*question_pairs)
del question_pairs
all_questions = question1+question2
print 'loaded {} in questions in total'.format(len(all_questions))



tagged_docs = [q.split() for q in all_questions]
tagged_docs = [TaggedDocument(q,[i]) for i,q in enumerate(tagged_docs)]
print 'loaded {} tagged documents'.format(len(tagged_docs))


model = Doc2Vec(dm=0,workers=psutil.cpu_count(),size=300)
model.build_vocab(tagged_docs)
words_vocab = model.wv.vocab
print 'built doc2vec vocab of len: ',len(words_vocab)

# word_vectors = load_pretrained_word_vectors(vocab=words_vocab)
# print 'loaded {} pretrained GloVe word vectors'.format(len(word_vectors))

# init_word_vectors_in_model (model,model.wv.index2word,word_vectors)
# print 'initialized the doc2vec model with pretrained word vectors'

model.train(tagged_docs, total_examples=len(all_questions), epochs=10)

model.save('doc2vec_model')

q1_indexes = xrange(len(all_questions)/2)
q1_vectors = {i:model.docvecs[i].tolist() for i in q1_indexes}
q2_indexes = xrange(len(all_questions)/2,len(all_questions))
q2_vectors = {i-len(question1):model.docvecs[i].tolist() for i in q2_indexes}

with open('../data/q1_docvecs.json','w') as fh:
    json.dump(obj=q1_vectors,fp=fh,indent=4)

with open('../data/q2_docvecs.json','w') as fh:
    json.dump(q2_vectors,fh,indent=4)

x = np.hstack((np.array(q1_vectors.values()),
               np.array(q2_vectors.values())))
svm_fit(x,labels)
# xgboost_fit(x,labels)
# mlp_fit(x,labels)



