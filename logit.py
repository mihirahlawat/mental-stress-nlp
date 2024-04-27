import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score, f1_score, precision_score
from embeddings import process_lines, load_doc, load_embedding, all_documents

SEED = 577

data = pd.read_csv('Suicide_Detection_cleaned-2.csv')

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

train_text, test_text, train_labels, test_labels = train_test_split(data['text'], data['class'], random_state=SEED,
                                                                    test_size=0.2, stratify=data['class'])

word2vec = load_embedding('embedding_word2vec.txt')

train_clean = process_lines(train_text, vocab)
test_clean = process_lines(test_text, vocab)
train_vec, train_labels_new = all_documents(train_clean, train_labels,word2vec)
test_vec, test_labels_new = all_documents(test_clean, test_labels, word2vec)

lr = LogisticRegression()
lr.fit(train_vec, train_labels_new)

y_test_pred = lr.predict(test_vec)
print('Test set accuracy %s' % accuracy_score(test_labels_new, y_test_pred))
print(classification_report(test_labels_new, y_test_pred))
