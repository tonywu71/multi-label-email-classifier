# Kaggle Competition
# 03/02/21
# Matthias Lesage

import json
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def clean_mail_type(etype):
    if type(etype) != str and np.isnan(etype):
        return 'text/plain'
    if (('multipart/' in etype or etype == 'text/calendar') and not(etype in ['multipart/alternative', 'multipart/mixed'])):
        return 'multipart/others'
    else:
        return etype.strip().lower()
    
def clean_date(date):
    # Remove occurences of GMT
    pre_date = date.split('GMT', 1)[0].split('(', 1)[0]
    
    # replace - with ' '
    pre_date = pre_date.replace('-', ' ')
    
    # remove multiple spaces
    pre_date = ' '.join(pre_date.split())
    
    # remove space at the beginning and the end
    words_list = pre_date.strip().split(' ')
    
    if words_list[0][:-1].isalpha():
        # We have the weekday
        words_list.pop(0)
    if len(words_list[0]) == 1:
        words_list[0] = '0' + words_list[0]
    assert len(words_list[0]) == 2, 'For date ' + date
    
    words_list[1] = words_list[1].capitalize()
    
    if len(words_list[2]) == 2:
        words_list[2] = '20' + words_list[2]
    assert len(words_list[2]) == 4, 'For date ' + date
              
    # Remove parenthesis
    if not(words_list[-1][0] in ['+', '-']):
        if words_list[-1][0] == '(':
            words_list.pop()

        # Add +0000 if not precised
        if not(words_list[-1][0] in ['+', '-']):
            if words_list[-1].isnumeric():
                words_list[-1] = '+' + words_list[-1]
            else:
                words_list.append('+0000')
        
    # Correct some errors
    if int(words_list[-1][3:]) > 60:
        words_list[-1] = str(int(words_list[-1][1:3])+1) + '00'
        if len(words_list[-1]) == 3:
            words_list[-1] = '+0' + words_list[-1]
        else:
            words_list[-1] = '+' + words_list[-1]
                
    assert int(words_list[-1][3:]) <= 60, 'For date ' + date + '. UTC offset : ' + words_list[-1]
    clean_date = ' '.join(''.join(map(str, word)) for word in words_list)
    assert len(clean_date) == 26, 'For date ' + date + '. Output : ' + clean_date + '. Len : ' + str(len(clean_date))
    return clean_date

def clean_timezone(date):
    # Remove occurences of GMT
    pre_date = date.split('GMT', 1)[0].split('(', 1)[0]
    
    # replace - with ' '
    pre_date = pre_date.replace('-', ' ')
    
    # remove multiple spaces
    pre_date = ' '.join(pre_date.split())
    
    # remove space at the beginning and the end
    words_list = pre_date.strip().split(' ')
    
    if words_list[0][:-1].isalpha():
        # We have the weekday
        words_list.pop(0)
    if len(words_list[0]) == 1:
        words_list[0] = '0' + words_list[0]
    assert len(words_list[0]) == 2, 'For date ' + date
    
    words_list[1] = words_list[1].capitalize()
    
    if len(words_list[2]) == 2:
        words_list[2] = '20' + words_list[2]
    assert len(words_list[2]) == 4, 'For date ' + date
              
    # Remove parenthesis
    if not(words_list[-1][0] in ['+', '-']):
        if words_list[-1][0] == '(':
            words_list.pop()

        # Add +0000 if not precised
        if not(words_list[-1][0] in ['+', '-']):
            if words_list[-1].isnumeric():
                words_list[-1] = '+' + words_list[-1]
            else:
                words_list.append('+0000')
        
    # Correct some errors
    if int(words_list[-1][3:]) > 60:
        words_list[-1] = str(int(words_list[-1][1:3])+1) + '00'
        if len(words_list[-1]) == 3:
            words_list[-1] = '+0' + words_list[-1]
        else:
            words_list[-1] = '+' + words_list[-1]
                
    assert int(words_list[-1][3:]) <= 60, 'For date ' + date + '. UTC offset : ' + words_list[-1]
    clean_date = words_list[-1]
    assert len(clean_date) == 5, 'For date ' + date + '. Output : ' + clean_date + '. Len : ' + str(len(clean_date))
    return clean_date[1:]

from sklearn.base import BaseEstimator, TransformerMixin
from datetime import date

class DateTransformer(BaseEstimator, TransformerMixin):
    """Transformer for date."""
    def __init__(self, active=True):
        self.active=active
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X.copy()
        if self.active:
            X_new['year'] = X['date'].dt.year
            X_new['month//2'] = X['date'].dt.month//2
            X_new['weekday'] = X['date'].dt.weekday
            X_new['trimester_from_2012'] = (X['date'].dt.date- date(2012,1,1)).dt.days//(7*13)
            X_new['hours'] = X['date'].dt.hour
        return X_new[['year', 'month//2', 'weekday', 'trimester_from_2012', 'hours']]

class BoundedLabelTransformer(BaseEstimator, TransformerMixin):
    """Transformer for org, tld and timezone."""
    def __init__(self, org=56, tld=23, timezone=8):
        self.org = org
        self.tld = tld
        self.timezone = timezone
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X[['org', 'tld', 'timezone']].copy()
        for column in ['org', 'tld', 'timezone']:
            if column == 'timezone':
                keep_labels = list(X[column].value_counts()[:self.timezone].index)
                X_new[column] = X_new[column].apply(lambda x : x if x in keep_labels else '0000')
            else:
                X_new[column] = X_new[column].fillna('unknown')
                if column == 'org':
                    keep_labels = list(X[column].value_counts()[:self.org].index)
                else:
                    keep_labels = list(X[column].value_counts()[:self.tld].index)
                X_new[column] = X_new[column].apply(lambda x : x if x in keep_labels else '__others')
        return X_new


class BoundedOrdinalTransformer(BaseEstimator, TransformerMixin):
    """Transformer for images and urls."""
    def __init__(self, images=10, urls=50):
        self.images = images
        self.urls = urls
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_new = X[['images', 'urls']].copy()
        
        for column in ['images', 'urls']:
            X_new[column] = X_new[column].fillna(0)
            if column == 'images':
                X_new[column] = X_new[column].apply(lambda x : x if x < self.images else self.images)
            else:
                X_new[column] = X_new[column].apply(lambda x : x if x < self.urls else self.urls) 
        
        #X_new['images/body'] = X['images']/X['chars_in_body']
        return X_new.values

def main(n_iter=200, n_jobs=-1, n_cv=5):
    """Principal function to run the computations"""

    # Preprocessing
    X_full = pd.read_csv('train_ml.csv', index_col=0)

    y_col = ['updates', 'personal', 'promotions', 'forums', 'purchases', 'travel', 'spam', 'social']
    y = X_full[y_col]
    X_full.drop(y_col, axis=1, inplace=True)

    X_full['timezone'] = X_full['date'].apply(clean_timezone)
    X_full['date'] = pd.to_datetime(X_full['date'].apply(clean_date), format="%d %b %Y %X %z", utc=True)
    X_full['mail_type'] = X_full['mail_type'].apply(clean_mail_type)
    X_full['images/body'] = X_full['images']/X_full['chars_in_body']
    X_full['salutations&designation'] = X_full['salutations'] & X_full['designation']
    
    date_cat = ['date']
    bounded_label_cat = ['org', 'tld', 'timezone']
    bounded_ordinal_cat = ['images', 'urls']
    binary_cat = ['ccs', 'bcced', 'salutations', 'designation', 'salutations&designation']
    label_cat = ['mail_type']
    continuous_cat = ['chars_in_subject', 'chars_in_body', 'images/body']

    Bounded_label_lin = make_pipeline(
        BoundedLabelTransformer(org=56, tld=23, timezone=8),
        OneHotEncoder(handle_unknown='ignore')
    )

    Label_lin = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='text/plain'),
        OneHotEncoder(handle_unknown='ignore')
    )

    Countinuous_lin = make_pipeline(
        SimpleImputer(strategy='mean'),
        RobustScaler()
    )

    processor_lin = make_column_transformer(
        (DateTransformer(active=True), date_cat),
        (Bounded_label_lin, bounded_label_cat),
        (BoundedOrdinalTransformer(images=10, urls=50), bounded_ordinal_cat),
        (Label_lin, label_cat),
        (Countinuous_lin, continuous_cat),
        (SimpleImputer(strategy='constant', fill_value=0), binary_cat)
    )

    param_grid = {
    'randomforestclassifier__n_estimators': [50,100,150,200,300,400,500]
    }

    labels = [
        'updates',
        'personal',
        'promotions',
        'forums',
        'purchases',
        'travel',
        'spam',
        'social'
    ]

    results_estimators = {}
    for label in labels:
        print("Training for", label, "...")
        clf = make_pipeline(processor_lin, RandomForestClassifier(random_state=1))
        y_class = y[label]
        
        grid_search=GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            scoring='neg_log_loss',
            cv=4,
            verbose=0,
            n_jobs=4,
            return_train_score=True,
            refit=False
        )

        start = time.time()
        grid_search.fit(X_full, y_class)
        end = time.time()
        print("Finished after", str(datetime.timedelta(seconds=(end-start))), "seconds.")

        results_estimators[label] = grid_search.cv_results_


    # Saving results in files
    for label in labels:
        parsed = json.loads(pd.DataFrame(results_estimators[label]).to_json())
        with open('results_'+ label + '.json', "w") as f:
                json.dump(parsed,f)
    
    print("------------")
    print("JSON generated!")
    print("------------")

    return 'Success'

if __name__=='__main__':
    main()
