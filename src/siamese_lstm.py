import keras
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import os
import numpy as np
from time import time
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from copy import deepcopy

stop_words = set(stopwords.words("english"))
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Merge, Embedding, Input
from keras.models import Model
import keras.backend as K
from keras.layers import LSTM, Bidirectional
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adadelta
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors

def tokenize_question(q, lower=False):
    try:
        tokens = word_tokenize(q)
    except UnicodeDecodeError:
        tokens = word_tokenize(q.decode('utf-8'))
    except:
        return ["<UNK>"]
    word_tokens = [word for word in tokens if word.isalpha()]
    word_tokens = [word for word in word_tokens if word not in stop_words]
    if (lower):
        word_tokens = map(lambda x: x.lower(), word_tokens)  # converting all to lower case
    return word_tokens


seq_length_list = []


def get_word_to_int_sequence(tokens):
    '''Returns sequence and updates vocab'''
    '''Does increasing number of functions impact performance?'''
    seq = []
    # global max_seq_length
    for token in tokens:
        if (token not in word_to_int):
            word_to_int[token] = len(word_to_int)
            int_to_word[len(word_to_int)] = token
            seq.append(word_to_int[token])
        else:
            seq.append(word_to_int[token])
            # if(len(seq)>max_seq_length):
            #   max_seq_length = len(seq)
    seq_length_list.append(len(seq))
    return seq


def process_dataset(dataset, dodeepcopy=True):
    '''Input is a numpy array of questions
    Output is an array of sequences according to word_to_int dict'''
    if (not dodeepcopy):
        dataset_mod = dataset
    else:
        dataset_mod = deepcopy(dataset)

    for i, row in enumerate(dataset_mod):
        # print(i) #for debugging
        q1, q2 = row[3], row[4]  # these correspond to the question
        q1_tokens, q2_tokens = tokenize_question(q1), tokenize_question(q2)
        q1_seq, q2_seq = get_word_to_int_sequence(q1_tokens), get_word_to_int_sequence(q2_tokens)
        row[3], row[4] = q1_seq, q2_seq

    return dataset_mod

# Hyperparameters and initial settings
embedding_dims = 300
hidden_dims = 64
gradient_clipping_norm = 1.25
n_epoch = 25
n_batch = 64

training_file = "../data/quora_duplicate_questions.tsv"
df_train = pd.read_csv(training_file,sep='\t')#[:100]

# All initial values, running this cell will reset all base variables
word_to_int = {"<UNK>": 0}
int_to_word = {0: "<UNK>"}
X = df_train.as_matrix()

vectors_dir = "../embeddings/gensim_format_glove.6B.300d.txt"


print("Processing dataset")
X = process_dataset(X, dodeepcopy=False)
print("Dataset processed")

max_seq_length = int(round(np.mean(seq_length_list) + 2 * np.std(seq_length_list)))

'''Load embeddings'''
print("Loading embeddings")
word_vectors = KeyedVectors.load_word2vec_format(vectors_dir, binary=False)
word_embeddings = np.random.randn(len(word_to_int), embedding_dims)
word_embeddings[0] = 0
for word, i in word_to_int.items():
    if (word in word_vectors.vocab):
        word_embeddings[i] = word_vectors.word_vec(word)
print("Embeddings loaded")

validation_size = int(round((0.1) * X.shape[0]))
training_size = X.shape[0] - validation_size
print("Validation size:{} and training size:{}".format(validation_size, training_size))

X_qs = np.array(map(lambda x: [x[3], x[4]], X))
Y = np.array(map(lambda x: [x[-1]], X))
X_train, X_val, Y_train, Y_val = train_test_split(X_qs, Y, test_size=validation_size)

pad_seq = lambda x: sequence.pad_sequences(x, maxlen=max_seq_length)

# Split to dicts
X_training = {'left': pad_seq(X_train[:, 0]), 'right': pad_seq(X_train[:, 1])}
X_validation = {'left': pad_seq(X_val[:, 0]), 'right': pad_seq(X_val[:, 1])}

'''Final model'''
def l1_dist_exp(left, right):
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))  # returns negative l1 norm exponent across batches


q1_placeholder = Input(shape=(max_seq_length,), dtype="int32")
q2_placeholder = Input(shape=(max_seq_length,), dtype="int32")
embed_layer = Embedding(len(word_embeddings), embedding_dims,
                        weights=[word_embeddings],
                        input_length=max_seq_length,
                        trainable=False)
# Experiment with trainable but large corpus should not affect

embedded_q1, embedded_q2 = embed_layer(q1_placeholder), embed_layer(q2_placeholder)

siamese_LSTM = LSTM(hidden_dims)
q1_enc, q2_enc = siamese_LSTM(embedded_q1), siamese_LSTM(embedded_q2)

l1_dist = Merge(mode=lambda x: l1_dist_exp(x[0], x[1]), output_shape=lambda x: (x[0][0], 1))([q1_enc, q2_enc])

final_model = Model(inputs=[q1_placeholder, q2_placeholder], outputs=[l1_dist])

optimizer = Adadelta(clipnorm=gradient_clipping_norm)
final_model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['accuracy'])

t1 = time()
trained_model = final_model.fit([X_training['left'], X_training['right']], Y_train, batch_size=n_batch,
                                nb_epoch=n_epoch,
                                validation_data=([X_validation['left'], X_validation['right']], Y_val))
print("Training finished.\n {}epochs in {}".format(n_epoch, time() - t1))

'''Plot accuracy'''
# Plot accuracy
plt.plot(trained_model.history['acc'])
plt.plot(trained_model.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig("accyracy.png")
plt.close()
# Plot loss

plt.plot(trained_model.history['loss'])
plt.plot(trained_model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig("loss.png")

'''Save model'''
model_json = trained_model.to_json()
with open("final_ma_lstm.json", 'w') as json_file:
    json_file.write(model_json)
trained_model.save_weights("final_ma_lstm.h5")