#!/usr/bin/env python
# coding: utf-8

# # install the required module
# 

# In[1]:


pip install jupyterthemes


# # Import data and Perform Exploratory Data Analysis

# ##### setting the style of the notebook to be monokai theme  
# 
# ##### If you don't run this code line, you will notice that the xlabel and ylabel on any plot is black on black and it will be hard to see them. 
# 

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 


# ### Load the data
# 

# In[3]:


reviews_df = pd.read_csv('amazon_reviews.csv')
reviews_df


# ### View the dataframe information

# In[4]:


reviews_df.info()


# ### view dataframe statistical summary

# In[5]:


reviews_df.describe()


# ### plot the count plot for the ratings

# In[6]:


sns.countplot(x = reviews_df['rating']) 


# ### get the length of the verified_reviews column

# In[7]:


reviews_df['length'] = reviews_df['verified_reviews'].apply(len)


# In[8]:


reviews_df


# ### plot the histogram for the length

# In[9]:


reviews_df['length'].plot(bins=100, kind='hist') 


# ### apply the describe method to get statistical summary

# In[10]:


reviews_df.describe()


# ### Plot the countplot for feedback
# ### Positive ~2800
# ### Negative ~250

# In[11]:


sns.countplot(x = reviews_df['feedback'])


# # PLOT WORDCLOUD

# ### Obtain only the positive reviews

# In[12]:


positive = reviews_df[reviews_df['feedback'] == 1]
positive


# ### Obtain the negative reviews only

# In[13]:


negative = reviews_df[reviews_df['feedback'] == 0]
negative


# ### convert to list format

# In[14]:


sentences = positive['verified_reviews'].tolist()
len(sentences)


# ### join all reviews into one large string

# In[16]:


sentences_as_one_string =" ".join(sentences)


# In[17]:


sentences_as_one_string


# In[19]:


pip install WordCloud


# In[20]:


from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[21]:


sentences = negative['verified_reviews'].tolist()
len(sentences)
sentences_as_one_string =" ".join(sentences)
plt.figure(figsize = (20,20))
plt.imshow(WordCloud().generate(sentences_as_one_string))


# In[25]:


import nltk # Natural Language tool kit 
nltk.download('stopwords')


# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# # Data Cleaning

# ### Let's define a pipeline to clean up all the messages 
# ### The pipeline performs the following: (1) remove punctuation, (2) remove stopwords
# 

# In[26]:


def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# ### Test the newly added function

# In[27]:


reviews_df_clean = reviews_df['verified_reviews'].apply(message_cleaning)


# ### show the original review

# In[28]:


print(reviews_df['verified_reviews'][5]) 


# ### show the original review

# In[29]:


print(reviews_df_clean[5])


# ### Define the cleaning pipeline we defined earlier

# In[30]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = message_cleaning)
reviews_countvectorizer = vectorizer.fit_transform(reviews_df['verified_reviews'])


# In[31]:


print(vectorizer.get_feature_names())


# In[32]:


print(reviews_countvectorizer.toarray())  


# In[33]:


reviews_countvectorizer.shape


# In[34]:


reviews = pd.DataFrame(reviews_countvectorizer.toarray())


# In[35]:


X = reviews


# In[36]:


y = reviews_df['feedback']
y


# # Train and Test AI/ML Models

# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[38]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[39]:


from sklearn.metrics import classification_report, confusion_matrix


# ### Predicting the Test set results

# In[40]:



y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[41]:


print(classification_report(y_test, y_predict_test))


# In[42]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))


# In[43]:


from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot = True)

print(classification_report(y_test, y_pred))


# In[ ]:




