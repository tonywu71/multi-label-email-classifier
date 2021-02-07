#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.base import BaseEstimator, ClassifierMixin

class ProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    '''Must be called at the very beginning of the pipeline'''

    def __init__(self, threshold_proba=0.1, threshold_number=5, cut=0.001):
        self.threshold_proba = threshold_proba
        self.threshold_number = threshold_number
        self.cut = cut

    def fit(self, X, y):
        '''Need dataframe as input...'''
        train_df = pd.concat([X_full,y], axis=1).dropna(axis=0, subset=['org'])
        self.labels = list(y.columns)
        
        y_proba = train_df.groupby(by='org').mean()[self.labels] # the probabilites to be predicted
        select_df = ((y_proba < self.threshold_proba) | (y_proba > (1-self.threshold_proba))) # 1 if it will be predicted
        number_df = train_df.groupby(by='org')[self.labels].count() > self.threshold_number # the number of each item is greater than the threshold_number?
        self.orgs = set(train_df.org.unique()) # the list of the organizations
        self.fit_df = y_proba*((select_df & number_df).applymap(lambda x: 1 if x else np.NaN))     
        return self

    def predict(self, X):
        results_df = pd.DataFrame(np.NaN, index=X.index, columns=self.labels)
        
        for idx, row in X.iterrows():
            if row['org'] in self.orgs:
                results_df.loc[idx] = self.fit_df.loc[row['org']]
        
        # Cutting
        results_df[results_df.notnull() & (results_df > (1-self.cut))] = 1-self.cut
        results_df[results_df.notnull() & (results_df < self.cut)] = self.cut
        
        return results_df


# In[22]:


class FinalClassifier(BaseEstimator, ClassifierMixin):
    '''Would it work good ?'''
    
    def __init__(self, prev_clf, threshold_proba=0.01, threshold_number=5, cut=0.0001):
        self.threshold_proba = threshold_proba
        self.threshold_number = threshold_number
        self.cut = cut
        self.prev_clf = prev_clf
        
    def fit(self, X, y):
        '''Need dataframe as input...'''
        self.clf = ProbabilisticClassifier(
            threshold_proba=self.threshold_proba,
            threshold_number=self.threshold_number,
            cut=self.cut
        )
        self.clf.fit(X,y)
        return self

    def predict(self, X):
        y_pred_proba_clf = self.clf.predict(X)
        
        y_pred_prev_clf = pd.DataFrame(
            data=self.prev_clf.predict_proba(X),
            columns=y_pred_proba_clf.columns,
            index=y_pred_proba_clf.index
        )
        
        results_df = y_pred_proba_clf.isnull()*y_pred_prev_clf + y_pred_proba_clf.fillna(0)
        
        return results_df.to_numpy()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, stratify=y, random_state=2)

# Comparing the new and the old algorithm
rfc = make_pipeline(processor_lin, OneVsRestClassifier(RandomForestClassifier(random_state=1), n_jobs=-1))
get_ipython().run_line_magic('time', 'rfc.fit(X_train,y_train)')

fcf = FinalClassifier(rfc, threshold_proba=0.01, threshold_number=5, cut=0.0001)
get_ipython().run_line_magic('time', 'fcf.fit(X_train,y_train)')

y_pred_charles = fcf.predict(X_test)
y_pred_rfc = rfc.predict_proba(X_test)

print("Log-loss for new clf", log_loss(y_test, y_pred_charles))
print("Log-loss for old clf", log_loss(y_test, y_pred_rfc))

