import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
import seaborn as sns

data = pd.read_csv('Suicide_Detection_cleaned.csv')
print(data.head())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['cleaned_text'])
word_freq = pd.DataFrame(tokenizer.word_counts.items(), columns=['word', 'count']).sort_values(by='count',
                                                                                               ascending=False)
# Plot bar graph for word frequency
plt.figure(figsize=(16, 8))
sns.barplot(x='count',y='word',data=word_freq.iloc[:30])
plt.title('Most Frequent Words')
plt.xlabel("Frequency")
plt.ylabel("Word")
plt.show()

data['cleaned_text'] = data['cleaned_text'].str.replace('filler', '')
data = data[data['cleaned_text'].apply(lambda x: len(x.split())!=0)]
data.reset_index(drop=True, inplace=True)
data.head()

posts_len = [len(x.split()) for x in data['cleaned_text']]
pd.Series(posts_len).hist(bins=60)
plt.show()
print(pd.Series(posts_len).describe())

print(data.shape)
datamid = data[data['cleaned_text'].apply(lambda x: len(x.split())<=82 and len(x.split())>=14)]
datamid.reset_index(drop=True, inplace=True)
print(datamid.shape)

print(datamid['class'].value_counts())

datamid.reset_index(drop=True, inplace=True)
datamid.replace({"class": {"SuicideWatch": 2, "depression": 1, "teenagers": 0}}, inplace=True)
datamid.drop(columns=['text'], inplace=True)
datamid = datamid.rename(columns={"cleaned_text": "text"})
datamid.head(10)

datamid.to_csv('Suicide_Detection_cleaned-2.csv', index=False)
