#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split


# In[2]:


df_train=pd.read_csv("train.csv")
df_train.head()
df_test=pd.read_csv("test.csv")


# In[3]:


df_train.shape


# In[4]:


df_train.columns=["polarity","title","text"]
df_test.columns=["polarity","title","text"]


# In[5]:


train=df_train
test=df_test


# In[6]:


train.drop(["title"],axis=1,inplace=True)
train=df_train.sample(500000,random_state=99)
train.head()


# In[51]:


train.polarity.value_counts()


# In[23]:


train.shape


# In[7]:


test.drop(["title"],axis=1,inplace=True)
test=df_test.sample(200000,random_state=99)
test.head()


# In[8]:


train['polarity'] = train['polarity'].apply(lambda x: 0 if x == 2 else 1)
test['polarity'] = test['polarity'].apply(lambda x: 0 if x == 2 else 1)
# 0 is positive
# 1 is negative


# In[9]:


print("train:\n", 
      train.isnull().sum(), 
      "\n", 
      "----------\n")
print("test:\n", 
      test.isnull().sum(), 
      "\n",
      "----------\n")


# In[10]:


len_train=len(train)
print(len_train)


# In[26]:


import nltk
import re
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
def preprocess_train_text(x):
    corpus=[]
    for i in range(0,500000):
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
    for i in range(0,200000):
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
    


# In[27]:


train_corpus=preprocess_train_text(train)
test_corpus=preprocess_test_text(test)


# In[28]:


train.shape
print(len(train_corpus))


# In[29]:


from collections import Counter
cnt = Counter()
for text in train_corpus:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(20) 


# In[15]:


from wordcloud import WordCloud
word_freq = dict(cnt)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[17]:


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


# In[19]:


from wordcloud import WordCloud
word_freq = dict(cnt)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[37]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1800)
X_train=cv.fit_transform(train_corpus)
y_train=train.iloc[:,0].values 
X_test=cv.transform(test_corpus)
y_test=test.iloc[:,0].values


# In[162]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
lr=LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_predict=lr.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
conf_matrix = confusion_matrix(y_test, y_predict)
class_report = classification_report(y_test, y_predict)

print("Accuracy             :", accuracy)
print("Confusion Matrix     :\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[161]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy             :", accuracy)
print("Confusion Matrix     :\n", conf_matrix)
print("Classification Report:\n", class_report)


# In[ ]:





# In[ ]:




