import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from time import time
from collections import Counter


# save list to file
def save_list(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def clean_line(line, vocab):
    tokens = line.split()
    tokens_clean = [w for w in tokens if w in vocab]
    return [tokens_clean]


# clean entire dataset
def process_lines(data, vocab):
    lines = list()
    for i in data:
        line = clean_line(i, vocab)
        lines += line
    return lines


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# load embedding as a dict
def load_embedding(filename):
    file = open(filename, 'r')
    lines = file.readlines()[1:]
    file.close()
    embedding = dict()
    for line in lines:
        parts = line.split()
        embedding[parts[0]] = np.asarray(parts[1:], dtype='float32')
    return embedding


def document_vector(doc, embeddings):
    sentence = list()
    """Create document vectors by averaging word vectors. Remove out-of-vocabulary words."""
    doc = [word for word in doc if word in embeddings.keys()]
    for i in doc:
        word = embeddings[i]
        sentence.append(word)
    return np.mean(sentence, axis=0)


# function for all the data
def all_documents(df, labels_ori, embeddings):
    vec = list()
    labels = list()
    for i in range(len(df)):
        if len(df[i]) == 0:
            continue
        else:
            vec.append(document_vector(df[i], embeddings))
            labels.append(labels_ori.values[i])
    return vec, labels


data = pd.read_csv('Suicide_Detection_cleaned-2.csv')

# Building Vocabulary
vocab = Counter()
tokens_list = [(s.split()) for s in data['text']]
for i in tokens_list:
    vocab.update(i)
min_occurance = 2
tokens = [k for k, c in vocab.items() if c >= min_occurance]
print(len(tokens))

# save tokens to a vocabulary file
save_list(vocab, 'vocab.txt')
vocabset = set(tokens)
cleantext = process_lines(data['text'], vocabset)

# set up the parameters of the model
model = Word2Vec(vector_size=300, window=10, min_count=1, seed=577)

t = time()
model.build_vocab(cleantext, progress_per=1000)
print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()
model.train(cleantext, total_examples=model.corpus_count, epochs=5, report_delay=1)
print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

# save model
filename = 'embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)

model.wv.most_similar('suicide')