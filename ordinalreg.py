import numpy as np
import pandas as pd
from embeddings import process_lines, load_doc, load_embedding, all_documents
from statsmodels.miscmodels.ordinal_model import OrderedModel

data = pd.read_csv('Suicide_Detection_cleaned-2.csv', header=0)
data.reset_index(drop=True, inplace=True)
data.replace({"class": {"SuicideWatch": 2, "depression": 1, "teenagers": 0}}, inplace=True)
data.drop(columns=['text'], inplace=True)
data = data.rename(columns={"cleaned_text": "text"})

vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)

word2vec = load_embedding('embedding_word2vec.txt')

text_clean = process_lines(data['text'], vocab)
text_vec, labels = all_documents(text_clean,data['class'],word2vec)

# Building an ordinal regression model
mod_log = OrderedModel(np.array(labels), np.array(text_vec), distr='logit')
res_log = mod_log.fit(method='bfgs', disp=False)
res_log.summary()