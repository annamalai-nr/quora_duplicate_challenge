# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import sys
import numpy as np
import networkx as nx
from graph_utils import (load_file,preprocessing,learn_model_and_predict_k_fold)
from nltk import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import accuracy_score, precision_score,recall_score,classification_report,f1_score,roc_auc_score

from utils import load_dataset

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def create_graphs_of_words(docs, window_size):
    """
    Create graphs of words

    """
    graphs = []
    sizes = []
    degs = []

    for doc in docs:
        words_list,pos_list = doc
        G = nx.Graph()
        for i in range(len(words_list)):
            if words_list[i] not in G.nodes():
                G.add_node(words_list[i])
            for j in range(i+1, i+window_size):
                if j < len(words_list):
                    G.add_edge(words_list[i], words_list[j])

        # for i in range(len(pos_list)):
        #     if pos_list[i] not in G.nodes():
        #         G.add_node(pos_list[i])
        #     for j in range(i+1, i+window_size):
        #         if j < len(pos_list):
        #             G.add_edge(pos_list[i], pos_list[j])

        graphs.append(G)
        sizes.append(G.number_of_nodes())
        try:
            degs.append(2.0*G.number_of_edges()/G.number_of_nodes())
        except:
            degs.append(0)

    print "Average number of nodes: ",np.mean(sizes)
    print "Average degree: ",np.mean(degs)

    return graphs


def spgk(sp_g1, sp_g2, norm1, norm2):
    """
    Compute spgk kernel

    """
    if norm1 == 0 or norm2==0:
        return 0
    else:
        kernel_value = 0
        for node1 in sp_g1:
            if node1 in sp_g2:
                kernel_value += 1
                for node2 in sp_g1[node1]:
                    if node2 != node1 and node2 in sp_g2[node1]:
                        kernel_value += (1.0/sp_g1[node1][node2]) * (1.0/sp_g2[node1][node2])

        kernel_value /= (norm1 * norm2)

        return kernel_value



def compute_spgraph_and_norm(graphs,depth):
    sp = []
    norm = []

    for g in graphs:
        current_sp = nx.all_pairs_dijkstra_path_length(g, cutoff=depth)
        sp.append(current_sp)

        sp_g = nx.Graph()
        for node in current_sp:
            for neighbor in current_sp[node]:
                if node == neighbor:
                    sp_g.add_edge(node, node, weight=1.0)
                else:
                    sp_g.add_edge(node, neighbor, weight=1.0 / current_sp[node][neighbor])

        M = nx.to_numpy_matrix(sp_g)
        norm.append(np.linalg.norm(M, 'fro'))
    return sp,norm

def build_kernel_matrix_for_pairs(graphs, depth):
    graphs1,graphs2 = zip(*graphs)
    sp1,norm1 = compute_spgraph_and_norm(graphs1,depth)
    sp2,norm2 = compute_spgraph_and_norm(graphs2,depth)

    kernel_values = []

    print "\nKernel computation progress:"
    last_len = 0
    for i in range(len(graphs)):
        sys.stdout.write('\b' * last_len)
        pct = 100 * (i+1) / len(graphs)
        out = '{}% [{}/{}]'.format(pct, i+1, len(graphs))
        last_len = len(out)
        sys.stdout.write(out)
        sys.stdout.flush()
        K = spgk(sp1[i], sp2[i], norm1[i], norm2[i])
        kernel_values.append(K)
    return kernel_values

def join_graphs(graphs1,graphs2):
    from networkx.algorithms.operators.binary import disjoint_union
    merged_graphs = []
    for g1,g2 in zip(graphs1,graphs2):
        g = disjoint_union(g1,g2)
        merged_graphs.append(g)
    return merged_graphs

def get_5_fold_cv_res(kernel_train,y_train,t,n_fold=5):
    accs = []
    fs = []
    for n in xrange(n_fold):
        x_,x_valid,y_,y_valid = train_test_split(kernel_train,
                                                   y_train,
                                                   test_size=0.8,
                                                   random_state=randint(0,100))
        del x_, y_
        y_pred = [1 if k >= t  else 0 for k in x_valid]
        accs.append(accuracy_score(y_valid,y_pred))
        fs.append(f1_score(y_valid,y_pred))

    # return np.array(accs).mean()
    return np.array(fs).mean()



def predict (kernel_values,labels,test_size=0.1,n_runs=5):
    acc = []
    p = []
    r = []
    f = []
    auc = []
    for i in xrange(n_runs):
        kernel_train,kernel_test,y_train,y_test = train_test_split(kernel_values,labels,
                                                                   test_size=test_size,
                                                                   random_state=randint(0,100))
        thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        accs = []
        for t in thresholds:
            avg_acc = get_5_fold_cv_res(kernel_train,y_train,t)
            accs.append(avg_acc)
        best_threshold = thresholds[accs.index(max(accs))]
        y_pred = [1 if k >= best_threshold  else 0 for k in kernel_test]
        acc.append(accuracy_score(y_test, y_pred))
        p.append(precision_score(y_test, y_pred))
        r.append(recall_score(y_test, y_pred))
        f.append(f1_score(y_test, y_pred))
        auc.append(roc_auc_score(y_test, y_pred))
        print 'run: {}, best threshold: {}'.format(i + 1,best_threshold)
        print classification_report(y_test, y_pred)

    from classify import print_mean_std
    mean_stds = print_mean_std(acc, p, r, f, auc)
    return mean_stds




def main():
    question_pairs, labels = load_dataset(load_n=None)
    labels = np.array(labels)
    q1s,q2s = zip(*question_pairs)
    window_size = 3
    depth = 2

    docs1 = [];pos1=[]
    for q in q1s:
        words = [w for w in word_tokenize(q)]
        pos_tags = nltk.pos_tag(words)
        try:
            words,pos_tags = zip(*[(w,pos) for w,pos in pos_tags if w.isalpha()])
        except:
            docs1.append(['invalid_question','XX'])
        docs1.append([stemmer.stem(w) for w in words])
        pos1.append(pos_tags)

    docs2 = [];pos2=[]
    for q in q2s:
        words = [w for w in word_tokenize(q)]
        pos_tags = nltk.pos_tag(words)
        try:
            words, pos_tags = zip(*[(w, pos) for w, pos in pos_tags if w.isalpha()])
        except:
            docs2.append(['invalid_question', 'XX'])
        docs2.append([stemmer.stem(w) for w in words])
        pos2.append(pos_tags)


    vocab1 = set([w for d in docs1 for w in d])
    vocab2 = set([w for d in docs2 for w in d])
    vocab = vocab1.union(vocab2)


    print "\nVocabulary size: ",len(vocab), vocab

    graphs1 = create_graphs_of_words(zip(docs1,pos1), window_size)
    graphs2 = create_graphs_of_words(zip(docs2,pos2), window_size)

    graphs = zip(graphs1,graphs2)
    kernel_values = build_kernel_matrix_for_pairs(graphs, depth)

    predict (kernel_values,labels)


if __name__ == "__main__":
    main()