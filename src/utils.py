import os
import csv
from pprint import pprint
import pandas as pd


def load_dataset(fname = '../data/quora_duplicate_questions.tsv',load_n=None):
    df = pd.DataFrame.from_csv(fname,sep='\t', encoding='utf8')
    df = df.fillna('')
    q1 = [q.lower() for q in df['question1'].values]
    q2 = [q.lower() for q in df['question2'].values]
    question_pairs = zip(q1,q2)[:load_n]
    labels = df['is_duplicate'][:load_n]

    return question_pairs, labels

if __name__ == '__main__':
    pass
