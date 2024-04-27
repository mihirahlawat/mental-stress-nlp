import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tensorflow.keras.preprocessing.text import Tokenizer
from wordcloud import WordCloud
from textblob import TextBlob
import nltk
from nltk.sentiment.util import *
nltk.download('punkt')


# Polarity score distribution
def get_polarity(text):
    return TextBlob(text).sentiment.polarity


df = pd.read_csv('Suicide_Detection_cleaned-2.csv', index_col=0)
df.reset_index(drop=True, inplace=True)
print(df.head())

print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))

sns.countplot(x=df['class'])
plt.title('Original Dataset Class Distribution')
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()

data_SuicideWatch = df[df['class'] == "SuicideWatch"]
data_depression = df[df['class'] == "depression"]
data_teenagers = df[df['class'] == "teenagers"]

# Wordcloud
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data_SuicideWatch['cleaned_text'])
word_freq_SuicideWatch = pd.DataFrame(tokenizer.word_counts.items(), columns=['word', 'count']).sort_values(by='count',
                                                                                                            ascending=False)
feature_names = word_freq_SuicideWatch['word'].values
wc = WordCloud(max_words=100, background_color="white", width=2000, height=1000)
wc.generate(' '.join(word for word in feature_names))
plt.figure(figsize=(20, 15))
plt.axis('off')
plt.imshow(wc)
plt.show()

# Get average text length
data_SuicideWatch['cleaned_text'] = data_SuicideWatch['cleaned_text'].astype('str')
data_SuicideWatch['length'] = data_SuicideWatch['cleaned_text'].apply(lambda x: len(x.split()))

# Plot average text length
ax = data_SuicideWatch['length'].plot(kind='hist', title='Distribution of Text Length - Suicidal Text', figsize=(8, 6))
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
plt.show()

data_SuicideWatch['Polarity'] = data_SuicideWatch['cleaned_text'].apply(get_polarity)
# Plot polarity score graph
ax = data_SuicideWatch['Polarity'].plot(kind='hist', title='Polarity Score - Suicidal Text', figsize=(8, 6))
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Count")
plt.show()

# Wordcloud
tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(data_depression['cleaned_text'])
word_freq_depression = pd.DataFrame(tokenizer2.word_counts.items(), columns=['word', 'count']).sort_values(by='count',
                                                                                                           ascending=False)
feature_names2 = word_freq_depression['word'].values
wc2 = WordCloud(max_words=100, background_color="white", width=2000, height=1000)
wc2.generate(' '.join(word for word in feature_names2))
plt.figure(figsize=(20, 15))
plt.axis('off')
plt.imshow(wc)
plt.show()

# Get average text length
data_depression['cleaned_text'] = data_depression['cleaned_text'].astype('str')
data_depression['length'] = data_depression['cleaned_text'].apply(lambda x: len(x.split()))

# Plot average text length
ax = data_depression['length'].plot(kind='hist', title='Distribution of Text Length - Depression Text', figsize=(8, 6))
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
plt.show()

data_depression['Polarity'] = data_depression['cleaned_text'].apply(get_polarity)
ax = data_depression['Polarity'].plot(kind='hist', title='Polarity Score - Depression Text', figsize=(8, 6))
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Count")
plt.show()

# Wordcloud
tokenizer3 = Tokenizer()
tokenizer3.fit_on_texts(data_teenagers['cleaned_text'])
word_freq_teenagers = pd.DataFrame(tokenizer3.word_counts.items(), columns=['word', 'count']).sort_values(by='count',
                                                                                                          ascending=False)
feature_names3 = word_freq_teenagers['word'].values
wc3 = WordCloud(max_words=100, background_color="white", width=2000, height=1000)
wc3.generate(' '.join(word for word in feature_names3))
plt.figure(figsize=(20, 15))
plt.axis('off')
plt.imshow(wc)
plt.show()

# Get average text length
data_teenagers['cleaned_text'] = data_teenagers['cleaned_text'].astype('str')
data_teenagers['length'] = data_teenagers['cleaned_text'].apply(lambda x: len(x.split()))

# Plot average text length
ax = data_teenagers['length'].plot(kind='hist', title='Distribution of Text Length - Teenager Text', figsize=(8, 6))
ax.set_xlabel("Number of Words")
ax.set_ylabel("Count")
plt.show()

data_teenagers['Polarity'] = data_teenagers['cleaned_text'].apply(get_polarity)
ax = data_teenagers['Polarity'].plot(kind='hist', title='Polarity Score - Teenager Text', figsize=(8, 6))
ax.set_xlabel("Polarity Score")
ax.set_ylabel("Count")
plt.show()

stats.ttest_ind(data_teenagers['Polarity'], data_depression['Polarity'], equal_var=False)
stats.ttest_ind(data_teenagers['Polarity'], data_SuicideWatch['Polarity'], equal_var=False)
stats.ttest_ind(data_depression['Polarity'], data_SuicideWatch['Polarity'], equal_var=False)
