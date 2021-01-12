#!/usr/bin/env python
# coding: utf-8

# <center> <font size=6> Kaggle challenge </font> </center>
# <center> <i> GroudTruth Team </i> </center>

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Import-modules" data-toc-modified-id="Import-modules-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Import modules</a></span></li><li><span><a href="#Import-data" data-toc-modified-id="Import-data-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import data</a></span></li><li><span><a href="#Information-about-the-dataset" data-toc-modified-id="Information-about-the-dataset-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Information about the dataset</a></span><ul class="toc-item"><li><span><a href="#Dataset-Features" data-toc-modified-id="Dataset-Features-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Dataset Features</a></span></li><li><span><a href="#Class-Labels-(8-Classes)" data-toc-modified-id="Class-Labels-(8-Classes)-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Class Labels (8 Classes)</a></span></li><li><span><a href="#Profile-Report" data-toc-modified-id="Profile-Report-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Profile Report</a></span></li></ul></li><li><span><a href="#Data-Analysis" data-toc-modified-id="Data-Analysis-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Analysis</a></span></li><li><span><a href="#Preprocessing" data-toc-modified-id="Preprocessing-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Preprocessing</a></span><ul class="toc-item"><li><span><a href="#NA-Cleaning" data-toc-modified-id="NA-Cleaning-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>NA Cleaning</a></span></li><li><span><a href="#Date" data-toc-modified-id="Date-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Date</a></span><ul class="toc-item"><li><span><a href="#Regex-parsing-for-date-extraction" data-toc-modified-id="Regex-parsing-for-date-extraction-5.2.1"><span class="toc-item-num">5.2.1&nbsp;&nbsp;</span>Regex parsing for date extraction</a></span></li><li><span><a href="#Applying-transformation-to-the-dataset" data-toc-modified-id="Applying-transformation-to-the-dataset-5.2.2"><span class="toc-item-num">5.2.2&nbsp;&nbsp;</span>Applying transformation to the dataset</a></span></li><li><span><a href="#Creating-date-features" data-toc-modified-id="Creating-date-features-5.2.3"><span class="toc-item-num">5.2.3&nbsp;&nbsp;</span>Creating date features</a></span></li></ul></li><li><span><a href="#Categorical-encoding" data-toc-modified-id="Categorical-encoding-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Categorical encoding</a></span><ul class="toc-item"><li><span><a href="#org,-tld-and-mail_types--&gt;-One-Hot-Encoding" data-toc-modified-id="org,-tld-and-mail_types-->-One-Hot-Encoding-5.3.1"><span class="toc-item-num">5.3.1&nbsp;&nbsp;</span>org, tld and mail_types -&gt; One-Hot Encoding</a></span></li><li><span><a href="#weekday--&gt;-One-Hot-Encoding" data-toc-modified-id="weekday-->-One-Hot-Encoding-5.3.2"><span class="toc-item-num">5.3.2&nbsp;&nbsp;</span>weekday -&gt; One-Hot Encoding</a></span></li></ul></li><li><span><a href="#Dropping-classes" data-toc-modified-id="Dropping-classes-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Dropping classes</a></span></li></ul></li><li><span><a href="#Final-dataset-overview" data-toc-modified-id="Final-dataset-overview-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Final dataset overview</a></span></li><li><span><a href="#Model" data-toc-modified-id="Model-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Model</a></span><ul class="toc-item"><li><span><a href="#For-one-class-first" data-toc-modified-id="For-one-class-first-7.1"><span class="toc-item-num">7.1&nbsp;&nbsp;</span>For one class first</a></span><ul class="toc-item"><li><span><a href="#Splitting-train-/-test" data-toc-modified-id="Splitting-train-/-test-7.1.1"><span class="toc-item-num">7.1.1&nbsp;&nbsp;</span>Splitting train / test</a></span></li><li><span><a href="#Prediction" data-toc-modified-id="Prediction-7.1.2"><span class="toc-item-num">7.1.2&nbsp;&nbsp;</span>Prediction</a></span></li></ul></li><li><span><a href="#GridSearch-for-one-class" data-toc-modified-id="GridSearch-for-one-class-7.2"><span class="toc-item-num">7.2&nbsp;&nbsp;</span>GridSearch for one class</a></span></li><li><span><a href="#For-all-classes" data-toc-modified-id="For-all-classes-7.3"><span class="toc-item-num">7.3&nbsp;&nbsp;</span>For all classes</a></span></li></ul></li><li><span><a href="#Generating-submissions-for-Kaggle" data-toc-modified-id="Generating-submissions-for-Kaggle-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Generating submissions for Kaggle</a></span><ul class="toc-item"><li><span><a href="#Pipeline-function" data-toc-modified-id="Pipeline-function-8.1"><span class="toc-item-num">8.1&nbsp;&nbsp;</span>Pipeline function</a></span></li></ul></li></ul></div>

# # Import modules

# In[1]:


import os
import json

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

import numpy as np

import pandas as pd
from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt
import seaborn as sns

import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, log_loss

import xgboost as xgb


# # Import data

# In[2]:


# os.listdir()


# In[3]:


df = pd.read_csv('train_ml.csv').drop(columns=['Unnamed: 0'])
df_test = pd.read_csv('test_ml.csv').drop(columns=['Unnamed: 0'])

# df.head()


# In[4]:


# df.info()


# # Information about the dataset

# ## Dataset Features
# 
# - date - unix style date format, date-time on which the email was received, e.g. Sat, 2 Jul 2016 11:02:58 +0530
# - org - organisation of the sender, e.g. centralesupelec, facebook, and google.
# - tld - top level domain of the organisation, eg. com, ac.in, fr, and org.
# - ccs - number of emails cced with this email, e.g. 0, 2, and 10.
# - bcced - is the receiver bcc'd in the email. Can take two values: 0 or 1.
# - mail_type - type of the mail body, e.g. text/plain and text/html.
# - images - number of images in the mail body, e.g. 0, 1, and 100.
# - urls - number of urls in the mail body, e.g. 0, 1, and 50.
# - salutations - is salutation used in the email? Either 0 or 1.
# - designation - is designation of the sender mentioned in the email. Either 0 or 1.
# - chars_in_subject - number of characters in the mail subject, e.g. 0, 1, and 10.
# - chars_in_body - number of characters in the mail body, e.g. 10 and 10000.
# - labels - last eight columns represent eight classes, 0 means that label is not present for this row and 1 means that label is present, multiple label columsn can be 1. Label columns are only present in train.csv. test.csv has features only.

# ## Class Labels (8 Classes)
# - updates
# - personal
# - promotions
# - forums
# - purchases
# - travel
# - spam
# - social

# In[5]:


# df.isnull().sum()


# ## Profile Report

# In[6]:


if False: # set to True to generate a new report
    profile = ProfileReport(df, title='Profiling Report')
    profile.to_file("profile_report.html")


# # Data Analysis

# In[7]:


df.columns


# In[8]:


list_classes = ['updates', 'personal', 'promotions', 'forums', 'purchases', 'travel',
       'spam', 'social']


# In[9]:


df[list_classes].idxmax(axis=1).value_counts()


# In[10]:


df[list_classes].idxmax(axis=1).value_counts().plot.bar();


# In[11]:


numerical_features = ["ccs", "images", "urls", "chars_in_subject", "chars_in_body"]
df[numerical_features]


# In[12]:


df[numerical_features].describe()


# In[13]:


fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharey=False)

for idx, (col, ax) in enumerate(zip(numerical_features, axs.ravel())):
    df[numerical_features].boxplot(col, ax=ax)


# In[14]:


df['class'] = df[list_classes].idxmax(axis=1)
# df.head()


# In[15]:


if False: # set to True to generate figure
    sns.pairplot(df[numerical_features + ['class']], hue='class');


# # Preprocessing

# ## NA Cleaning

# In[16]:


# len(df)


# In[17]:


df.isna().sum()


# In[18]:


# len(df.dropna(how='any'))


# In[19]:


# len(df.dropna(how='any')) / len(df)


# In[20]:


df = df.dropna(how='any')
# df.isna().sum()


# ## Date

# ### Regex parsing for date extraction

# For one example:

# In[21]:


string = "Mon, 15 Oct 2018 08:03:09 +0000 (UTC)"
pattern = r'(\d{1,2}.*\d{2}:\d{2}:\d{2}) ([+-]\d{2}\d{2})'
ans = re.search(pattern, string)


# In[22]:


date = ''.join([ans.group(1), ans.group(2)])
# date


# In[23]:


# pd.to_datetime(date)


# ### Applying transformation to the dataset

# Now let's apply the transformation to the dataset:

# In[24]:


def format_date(row):
    pattern = r'(\d{1,2}.*\d{2}:\d{2}:\d{2}) ([+-]\d{2}\d{2})'
    
    ans = re.search(pattern, string=row)
    
    if ans:
        if ans.group(1)[1] == ' ':
            return ''.join(['0', ans.group(1), ans.group(2)])
        else:
            return ''.join([ans.group(1), ans.group(2)])
    else:
        return np.nan


# In[25]:


df['date'] = pd.to_datetime(df['date'].apply(format_date), utc=True)
# df['date']


# In[26]:


# df['date'].iloc[0]


# In[27]:


# df.head()


# ### Creating date features

# In[28]:


df['date_day'] = df['date'].dt.date
# df['date_day']


# In[29]:


# df['date_day'].value_counts()


# In[30]:


# plt.figure(figsize=(20,12))
# df['date_day'].value_counts().head(40).plot.bar();


# In[31]:


df['month'] = df['date'].dt.month
# df['month']


# In[32]:


df['weekday'] = df['date'].dt.weekday
# df['weekday']


# In[33]:


df['hour'] = df['date'].dt.hour
# df['hour']


# In[34]:


# df.info()


# ## Categorical encoding

# ### org, tld and mail_types -> One-Hot Encoding

# After having tried **target encoding**, we'll now try one-hot encoding for all categorical features.

# In[35]:


# Example with mail_types:
# pd.get_dummies(df['mail_type'], prefix='mail_type_').head(3)


# In[36]:


list_categorical_cols = ["org", "tld", "mail_type"]

for col in list_categorical_cols:
    df = pd.concat([df, pd.get_dummies(df[col], prefix=f'{col}_')], axis=1).drop(columns=[col])

# df.head(3)


# ### weekday -> One-Hot Encoding

# In[37]:


# df['weekday'].value_counts()


# In[38]:


# pd.get_dummies(df['weekday'], prefix='weekday_').head(3)


# In[39]:


df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='weekday')], axis=1).drop(columns=['weekday'])
# df.head(3)


# ## Dropping classes

# In[40]:


cols_to_drop = ["date", "date_day", "class"]

df = df.drop(columns=cols_to_drop)


# # Final dataset overview

# In[41]:


# df.head(3).T


# In[42]:


# df.info()


# # Model

# ## For one class first

# In[63]:


k = 0
print(f"Current class: {list_classes[k]}")


# ### Splitting train / test

# In[64]:


X = df.drop(columns=list_classes)
y = df[list_classes[k]]


# In[65]:


# y


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)


# Quick check:

# In[67]:


# y_test.sum(axis=0) / (y_train.sum(axis=0) + y_test.sum(axis=0))


# ### Prediction

# In[48]:


model = xgb.XGBClassifier(n_estimators=100, random_state=0)


# model.fit(X_train, y_train, verbose=0)


# In[49]:


# y_preds = model.predict_proba(X_test)


# In[50]:


# log_loss(y_test, y_preds)


# In[51]:


# print(classification_report(y_test, model.predict(X_test)))


# ## GridSearch for one class

# In[69]:


if True:
    # A parameter grid for XGBoost
    params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5]
            }

    # It's more efficient to use parallelization for GridSearch so we set n_jobs=1:
    clf = xgb.XGBClassifier(n_estimators=100, n_jobs=-1, verbosity=0) 

    grid = GridSearchCV(
        clf,
        params,
        n_jobs=1,  # -1 means using all processors
        scoring="neg_log_loss",
        cv=3,
        refit=True,
        verbose=1)

    grid.fit(X, y, verbose=0)

    model = grid.best_estimator_

    # Save the optimal hyperparameters
    with open('XGBClassifier_optimal_parameters.json', "w") as f:  
        json.dump(grid.best_params_, f)


# In case we haven't recomputed GridSearch, we'll load the last saved model:

# In[53]:


with open('XGBClassifier_optimal_parameters.json') as f:
    params_optimal = json.load(f)

# params_optimal


# In[54]:


# model = xgb.XGBClassifier(random_state=0)
# model.set_params(**params_optimal)

# model.fit(X_train, y_train, verbose=0)

# y_preds = model.predict_proba(X_test)
# print(f"logloss: {log_loss(y_test, y_preds)}")
# print("\n")
# print(classification_report(y_test, model.predict(X_test)))


# ## For all classes

# In[58]:

print("------------- Fitting models: -------------")

dic_preds = dict()


pbar = tqdm(list_classes)

for current_class in pbar:
    pbar.set_description(f'Processing "{current_class}" class"')
    
    X = df.drop(columns=list_classes)
    y = df[current_class]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y)
    
    model = xgb.XGBClassifier(random_state=0)
    model.set_params(**params_optimal)
    
    model.fit(X_train, y_train, verbose=False, eval_metric='logloss')
    model.save_model(f"models/{current_class}_clf.model")
    
    dic_preds[current_class] = model.predict_proba(X_test)
    
    print(f"logloss: {log_loss(y_test, dic_preds[current_class])}")
    print("\n")
    
    print(f'For "{current_class}" class:')
    print(classification_report(y_test, model.predict(X_test)))
    
    print("-------------------------------------------------------------------\n")


# # Generating submissions for Kaggle

# ## Pipeline function

# In[59]:


def format_date(row):
    pattern = r'(\d{1,2}.*\d{2}:\d{2}:\d{2}) ([+-]\d{2}\d{2})'
    
    ans = re.search(pattern, string=row)
    
    if ans:
        if ans.group(1)[1] == ' ':
            return ''.join(['0', ans.group(1), ans.group(2)])
        else:
            return ''.join([ans.group(1), ans.group(2)])
    else:
        return np.nan

    
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


    

    
def proprocess(df):
    df_train = pd.read_csv("train_ml.csv")
    
    # Drop NA only for training set since we'll assume there is no NA is test set
    df_train = df_train.dropna(how='any')
    
    
    list_df = [df_train, df]
    
    
    for idx, cur_df in enumerate(list_df):
        ## Date features
        list_df[idx]['date'] = pd.to_datetime(list_df[idx]['date'].apply(format_date), utc=True)
        
        list_df[idx]['date_day'] = list_df[idx]['date'].dt.date
        list_df[idx]['month'] = list_df[idx]['date'].dt.month
        list_df[idx]['weekday'] = list_df[idx]['date'].dt.weekday
        list_df[idx]['hour'] = list_df[idx]['date'].dt.hour
    
    # Update DataFrames
    df_train = list_df[0]
    df = list_df[1]
    
    
    ## Categorical features to one-hot encode:
    list_categorical_cols = ["org", "tld", "mail_type"]

    for col in list_categorical_cols:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=f'{col}_')], axis=1).drop(columns=[col])
    
    
    
    ## weekday -> One-Hot Encoding
    df = pd.concat([df, pd.get_dummies(df['weekday'], prefix='weekday')], axis=1).drop(columns=['weekday'])
    
    
    ## Dropping classes
    cols_to_drop = ["date", "date_day"]
    df = df.drop(columns=cols_to_drop)
    
    
    ## Loading optimal hyperparameters
    with open('XGBClassifier_optimal_parameters.json') as f:
        params_optimal = json.load(f)
    
    
    ## Predictions
    dic_preds = dict()


    pbar = tqdm(list_classes)

    for current_class in pbar:
        pbar.set_description(f'Processing "{current_class}" class"')

        X = df # Here there is no labels at hand...

        model = xgb.XGBClassifier(random_state=0)
        model.set_params(**params_optimal)

        model.load_model(f"models/{current_class}_clf.model")
        
        # We only keep the probability of the positive classification:
        dic_preds[current_class] = model.predict_proba(X)[:,1]
    
    
    ans = pd.DataFrame(dic_preds)
    
    return ans


# In[60]:


df_test = pd.read_csv("test_ml.csv")
ans = proprocess(df_test)

ans.to_csv('submission_tony.csv', index_label='Id')
ans.head()


# In[61]:


pd.read_csv('submission_tony.csv')


# In[ ]:




