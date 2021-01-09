"""
This script can be used as skelton code to read the challenge train and test
csvs, to train a trivial model, and write data to the submission file.
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

## Read csvs

train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)

## Filtering column "mail_type"
train_x = train_df[['mail_type']]
train_x = train_x.fillna(value='None')
train_y = train_df[['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social']]

test_x = test_df[['mail_type']]
test_x = test_x.fillna(value='None')

## Do one hot encoding of categorical feature
feat_enc = OneHotEncoder()
feat_enc.fit(np.vstack([train_x, test_x]))
train_x_featurized = feat_enc.transform(train_x)
test_x_featurized = feat_enc.transform(test_x)

## Train a simple OnveVsRestClassifier using featurized data
classif = OneVsRestClassifier(SVC(kernel='linear', probability=True))
classif.fit(train_x_featurized, train_y)
pred_y = classif.predict_proba(test_x_featurized)
print (pred_y.shape)

## Save results to submission file
pred_df = pd.DataFrame(pred_y, columns=['updates', 'personal', 'promotions',
                        'forums', 'purchases', 'travel',
                        'spam', 'social'])
pred_df.to_csv("knn_sample_submission_ml.csv", index=True, index_label='Id')
