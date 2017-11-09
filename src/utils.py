import os
import csv
from pprint import pprint
import pandas as pd


def load_dataset(fname = '../data/quora_duplicate_questions.tsv'):
    df = pd.DataFrame.from_csv(fname,sep='\t', encoding='utf8')
    df = df.fillna('')
    question_pairs = zip(df['question1'].values,df['question2'].values)
    labels = df['is_duplicate']

    return question_pairs, labels

if __name__ == '__main__':
    pass
