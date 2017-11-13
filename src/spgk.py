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


def create_graphs_of_words(docs, window_size):
    """
    Create graphs of words

    """
    graphs = []
    sizes = []
    degs = []

    for doc in docs:
        G = nx.Graph()
        for i in range(len(doc)):
            if doc[i] not in G.nodes():
                G.add_node(doc[i])
            for j in range(i+1, i+window_size):
                if j < len(doc):
                    G.add_edge(doc[i], doc[j])

        graphs.append(G)
        sizes.append(G.number_of_nodes())
        degs.append(2.0*G.number_of_edges()/G.number_of_nodes())

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


def build_kernel_matrix(graphs, depth):
    """
    Build kernel matrices

    """
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
                    sp_g.add_edge(node, neighbor, weight=1.0/current_sp[node][neighbor])

        M = nx.to_numpy_matrix(sp_g)
        norm.append(np.linalg.norm(M,'fro'))

    K = np.zeros((len(graphs), len(graphs)))

    print "\nKernel computation progress:"
    last_len = 0
    for i in range(len(graphs)):
        sys.stdout.write('\b' * last_len)
        pct = 100 * (i+1) / len(graphs)
        out = '{}% [{}/{}]'.format(pct, i+1, len(graphs))
        last_len = len(out)
        sys.stdout.write(out)
        sys.stdout.flush()
        for j in range(i,len(graphs)):
            K[i,j] = spgk(sp[i], sp[j], norm[i], norm[j])
            K[j,i] = K[i,j]

    return K


def main():
    from utils import load_dataset
    question_pairs, labels = load_dataset(load_n=10000)
    labels = np.array(labels)
    docs = [q1+' '+q2 for q1,q2 in question_pairs]
    window_size = 2
    depth = 1

    vocab = set([word for q in docs for word in word_tokenize(q)])
    docs = [word_tokenize(d) for d in docs]

    print "\nVocabulary size: ",len(vocab), vocab

    graphs = create_graphs_of_words(docs, window_size)
    K = build_kernel_matrix(graphs, depth)

    learn_model_and_predict_k_fold(K, labels)


if __name__ == "__main__":
    main()