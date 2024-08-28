#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# In[5]:


df_train=pd.read_csv("train.csv")
df_train.head()
df_test=pd.read_csv("test.csv")


# In[6]:


df_train.columns=["polarity","title","text"]
df_test.columns=["polarity","title","text"]


# In[7]:


train=df_train
test=df_test


# In[12]:


# train.drop(["title"],axis=1,inplace=True)
train=df_train.sample(100000,random_state=99)
train.head()


# In[13]:


# test.drop(["title"],axis=1,inplace=True)
test=df_test.sample(100000,random_state=99)
test.head()


# In[14]:


train['polarity'] = train['polarity'].apply(lambda x: 0 if x == 2 else 1)
test['polarity'] = test['polarity'].apply(lambda x: 0 if x == 2 else 1)
# 0 is positive
# 1 is negative


# In[18]:


def clean_text(df, field):
    df[field] = df[field].str.replace(r"@"," at ")
    df[field] = df[field].str.replace("#[^a-zA-Z0-9_]+"," ")
    df[field] = df[field].str.replace(r"[^a-zA-Z(),\"'\n_]"," ")
    df[field] = df[field].str.replace(r"http\S+","")
    df[field] = df[field].str.lower()
    return df

clean_text(train,"text")
clean_text(test,"text")


# In[21]:


import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def preprocess_train_text(x):
    corpus=[]
    for i in range(0,100000):
        review=re.sub("^a-zA-Z"," ",x.iloc[i]["text"])
        review=review.lower()
        review=review.split()
        ps=PorterStemmer()
        all_stopwords=stopwords.words("English")
        all_stopwords.remove("not")
        review=[ps.stem(word) for word in review if not word in set(all_stopwords)] #remove words which wont help us 
        review=" ".join(review)
        corpus.append(review)
        
    return corpus
    
def preprocess_test_text(x):
    corpus=[]
    for i in range(0,100000):
        review=re.sub("^a-zA-Z"," ",x.iloc[i]["text"])
        review=review.lower()
        review=review.split()
        ps=PorterStemmer()
        all_stopwords=stopwords.words("English")
        all_stopwords.remove("not")
        review=[ps.stem(word) for word in review if not word in set(all_stopwords)] #remove words which wont help us 
        review=" ".join(review)
        corpus.append(review)
        
    return corpus
    


# In[23]:


train_corpus=preprocess_train_text(train)
test_corpus=preprocess_test_text(test)


# In[24]:


from collections import Counter
cnt = Counter()
for text in train_corpus:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(20) 


# In[25]:


from wordcloud import WordCloud
word_freq = dict(cnt)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[28]:


def remove_freqwords(text):
    return " ".join([word for word in str(text).split() if word not in {'book','would','read','it.'}])

train_corpus = [remove_freqwords(text) for text in train_corpus]
test_corpus = [remove_freqwords(text) for text in test_corpus]


# In[30]:


from collections import Counter
cnt = Counter()
for text in train_corpus:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(20) 


# In[31]:


from wordcloud import WordCloud
word_freq = dict(cnt)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[36]:


text_len = pd.Series([len(review.split()) for review in train_corpus])
text_len.plot(kind="box")
plt.ylabel("Text Length")


# In[56]:


import matplotlib.pyplot as plt

review_lengths = [len(review.split()) for review in train_corpus]
plt.hist(review_lengths, bins=50)
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.title('Distribution of Review Lengths')
plt.show()


# In[57]:


max_len = int(np.percentile(text_len, 95))
max_features = 1800
print(max_len)


# In[49]:


X_train=train_corpus
X_test=test_corpus
y_train=train.iloc[:,0].values
y_test=test.iloc[:,0].values


# In[40]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[41]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_r)


# In[43]:


# using tokenizer to transform text messages into training and testing set
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)


# In[44]:


X_train_seq_padded = pad_sequences(X_train_seq, maxlen=83)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=83)


# In[45]:


X_train_seq_padded = pad_sequences(X_train_seq, maxlen=64)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=64)


# In[46]:


BATCH_SIZE = 64

model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1,64))
model.add(Bidirectional(LSTM(100, dropout=0,recurrent_dropout=0)))
model.add(Dense(128, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile("adam","binary_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor="val_loss",patience=5,verbose=True)


# In[50]:


model_rnn = model.fit(X_train_seq_padded, y_train,batch_size=BATCH_SIZE,epochs=15,
                    validation_data=(X_test_seq_padded, y_test),callbacks=[early_stop])


# In[51]:


y_predicted_rnn_1=model.predict(X_test_seq_padded)


# In[58]:


from sklearn.metrics import roc_auc_score
pred_train = model.predict(X_train_seq_padded)
pred_test = model.predict(X_test_seq_padded)
print('LSTM Recurrent Neural Network baseline: ' + str(roc_auc_score(y_train, pred_train)))
print('LSTM Recurrent Neural Network: ' + str(roc_auc_score(y_test, pred_test)))


# In[59]:


model.evaluate(X_test_seq_padded, y_test)


# In[65]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from tabulate import tabulate

max_length = 100

predictions =model.predict(X_test_seq_padded)

predicted_labels = np.round(predictions)


accuracy = accuracy_score(y_test, predicted_labels)
precision = precision_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)
f1 = f1_score(y_test, predicted_labels)

table = [
    ["Accuracy", accuracy],
    ["Precision", precision],
    ["Recall", recall],
    ["F1-score", f1]
]
print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))


# In[66]:


# Assuming y_pred contains the predicted probabilities
y_pred_binary = (predictions > 0.5).astype(int)  # Set a threshold of 0.5

# Now you can compare y_pred_binary with y_test
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

class_report = classification_report(y_test, y_pred_binary)

print("Classification Report:\n", class_report)


# In[97]:


sample_text = "very bad "
sample_seq = tokenizer.texts_to_sequences([sample_text])[0]
sample_seq_padded = pad_sequences([sample_seq], maxlen=83 ,padding='post')
prediction = model.predict(sample_seq_padded)[0][0]  
if prediction > 0.5:
    predicted_class = 0  #+ve
else:
    predicted_class = 1 #-ve

print("Predicted class:", predicted_class)


# In[98]:


model.wv.most_similar("bad")


# In[ ]:




