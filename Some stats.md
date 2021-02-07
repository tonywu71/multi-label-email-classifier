LGBM

params = {'num_leaves': 300, "verbose":-1, "objective":"binary", "bagging_fraction":0.9, "bagging_freq":2, "feature_fraction":0.5, "metric":"logloss", "feature_fraction_bynode":0.5}
num_round = 90

tlds 2 orgs 2

```
For : updates ...
              precision    recall  f1-score   support

           0       0.95      0.95      0.95     12648
           1       0.91      0.91      0.91      7188

    accuracy                           0.93     19836
   macro avg       0.93      0.93      0.93     19836
weighted avg       0.93      0.93      0.93     19836

Train :  0.03422614988172397
Test :  0.16794369916126015 

For : personal ...
              precision    recall  f1-score   support

           0       0.96      0.92      0.94      3776
           1       0.98      0.99      0.99     16060

    accuracy                           0.98     19836
   macro avg       0.97      0.96      0.96     19836
weighted avg       0.98      0.98      0.98     19836

Train :  0.0071203226095515855
Test :  0.06863563666338539 

For : promotions ...
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     15874
           1       0.89      0.84      0.87      3962

    accuracy                           0.95     19836
   macro avg       0.93      0.91      0.92     19836
weighted avg       0.95      0.95      0.95     19836

Train :  0.01716730940237357
Test :  0.12974852856543567 

For : forums ...
              precision    recall  f1-score   support

           0       0.98      0.98      0.98     16745
           1       0.89      0.87      0.88      3091

    accuracy                           0.96     19836
   macro avg       0.93      0.93      0.93     19836
weighted avg       0.96      0.96      0.96     19836

Train :  0.01250480334822104
Test :  0.09109817179844701 

For : purchases ...
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19672
           1       0.92      0.55      0.69       164

    accuracy                           1.00     19836
   macro avg       0.96      0.78      0.84     19836
weighted avg       1.00      1.00      1.00     19836

Train :  0.00010515139767633277
Test :  0.023683992946498436 

For : travel ...
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19786
           1       0.68      0.52      0.59        50

    accuracy                           1.00     19836
   macro avg       0.84      0.76      0.79     19836
weighted avg       1.00      1.00      1.00     19836

Train :  5.005606006717306e-06
Test :  0.016707878946597265 

For : spam ...
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19761
           1       0.85      0.53      0.66        75

    accuracy                           1.00     19836
   macro avg       0.92      0.77      0.83     19836
weighted avg       1.00      1.00      1.00     19836

Train :  6.3703700594975615e-06
Test :  0.010610061375678746 

For : social ...
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     17834
           1       0.97      0.97      0.97      2002

    accuracy                           0.99     19836
   macro avg       0.99      0.99      0.99     19836
weighted avg       0.99      0.99      0.99     19836

Train :  0.00046123487616209083
Test :  0.01431927732890075 


 
Mean loss =  0.06534340584827542
```

For {'n_estimators': range(150,251,50), 'criterion': ['entropy'], 'max_features':[150,200,300]} 	50% size
2 orgs			3 tlds 				result=0.051

```
For : updates ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 12.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=200, n_estimators=150)
              precision    recall  f1-score   support

           0       0.95      0.95      0.95     12648
           1       0.92      0.91      0.91      7188

    accuracy                           0.94     19836
   macro avg       0.93      0.93      0.93     19836
weighted avg       0.94      0.94      0.94     19836

0.17560029465428476
For : personal ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 11.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=300, n_estimators=250)
              precision    recall  f1-score   support

           0       0.95      0.93      0.94      3776
           1       0.98      0.99      0.99     16060

    accuracy                           0.98     19836
   macro avg       0.97      0.96      0.96     19836
weighted avg       0.98      0.98      0.98     19836

0.09751078667661572
For : promotions ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 11.0min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=300, n_estimators=150)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     15874
           1       0.90      0.84      0.87      3962

    accuracy                           0.95     19836
   macro avg       0.93      0.91      0.92     19836
weighted avg       0.95      0.95      0.95     19836

0.13135357368962877
For : forums ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 19.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=200, n_estimators=250)
              precision    recall  f1-score   support

           0       0.98      0.98      0.98     16745
           1       0.89      0.87      0.88      3091

    accuracy                           0.96     19836
   macro avg       0.93      0.93      0.93     19836
weighted avg       0.96      0.96      0.96     19836

0.10020697141622371
For : purchases ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 11.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=300, n_estimators=250)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19672
           1       0.93      0.68      0.78       164

    accuracy                           1.00     19836
   macro avg       0.97      0.84      0.89     19836
weighted avg       1.00      1.00      1.00     19836

0.014374966192444367
For : travel ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  7.1min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=200, n_estimators=200)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19786
           1       0.90      0.52      0.66        50

    accuracy                           1.00     19836
   macro avg       0.95      0.76      0.83     19836
weighted avg       1.00      1.00      1.00     19836

0.01107627436127544
For : spam ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  5.0min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=200, n_estimators=200)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19761
           1       0.96      0.72      0.82        75

    accuracy                           1.00     19836
   macro avg       0.98      0.86      0.91     19836
weighted avg       1.00      1.00      1.00     19836

0.005411149625681329
For : social ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed: 13.6min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=300, n_estimators=200)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     17834
           1       0.98      0.99      0.98      2002

    accuracy                           1.00     19836
   macro avg       0.99      0.99      0.99     19836
weighted avg       1.00      1.00      1.00     19836

0.01200909328875456

 
Mean loss =  0.06844288873811359
```





For (mistake) {'n_estimators': range(150,251,50), 'criterion': ['entropy'], 'max_features':[250,400,600]} 50% size, result = 0.061 		2 orgs			3 tlds

```
For : updates ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  6.9min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=250, n_estimators=250)
              precision    recall  f1-score   support

           0       0.94      0.95      0.95     12648
           1       0.91      0.90      0.90      7188

    accuracy                           0.93     19836
   macro avg       0.93      0.92      0.93     19836
weighted avg       0.93      0.93      0.93     19836

0.20025152494426435
For : personal ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  5.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=250, n_estimators=150)
              precision    recall  f1-score   support

           0       0.95      0.91      0.93      3776
           1       0.98      0.99      0.98     16060

    accuracy                           0.97     19836
   macro avg       0.96      0.95      0.95     19836
weighted avg       0.97      0.97      0.97     19836

0.08701014468862008
For : promotions ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  6.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=400, n_estimators=250)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97     15874
           1       0.91      0.82      0.86      3962

    accuracy                           0.95     19836
   macro avg       0.93      0.90      0.91     19836
weighted avg       0.95      0.95      0.95     19836

0.1420328000521081
For : forums ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  5.7min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=250, n_estimators=250)
              precision    recall  f1-score   support

           0       0.97      0.98      0.98     16745
           1       0.90      0.83      0.87      3091

    accuracy                           0.96     19836
   macro avg       0.93      0.91      0.92     19836
weighted avg       0.96      0.96      0.96     19836

0.10721562391813233
For : purchases ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  3.7min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=400, n_estimators=150)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19672
           1       0.88      0.49      0.63       164

    accuracy                           1.00     19836
   macro avg       0.94      0.74      0.81     19836
weighted avg       0.99      1.00      0.99     19836

0.021792444380893145
For : travel ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  3.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=600, n_estimators=150)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19786
           1       1.00      0.42      0.59        50

    accuracy                           1.00     19836
   macro avg       1.00      0.71      0.80     19836
weighted avg       1.00      1.00      1.00     19836

0.015277191783122802
For : spam ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  2.6min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=400, n_estimators=150)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     19761
           1       0.86      0.67      0.75        75

    accuracy                           1.00     19836
   macro avg       0.93      0.83      0.88     19836
weighted avg       1.00      1.00      1.00     19836

0.004402471612978288
For : social ...
Fitting 3 folds for each of 9 candidates, totalling 27 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  27 out of  27 | elapsed:  4.6min finished
Best classifier :  RandomForestClassifier(criterion='entropy', max_features=400, n_estimators=200)
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     17834
           1       0.98      0.94      0.96      2002

    accuracy                           0.99     19836
   macro avg       0.98      0.97      0.98     19836
weighted avg       0.99      0.99      0.99     19836

0.03299659506882832

 
Mean loss =  0.07637234955611842
```





For {'n_estimators': range(100,401,100), 'criterion': ['entropy']} RF size 0.3 , decimation 20

```
For : updates ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  2.6min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=300)
              precision    recall  f1-score   support

           0       0.95      0.96      0.95      7589
           1       0.92      0.91      0.92      4313

    accuracy                           0.94     11902
   macro avg       0.94      0.93      0.94     11902
weighted avg       0.94      0.94      0.94     11902

0.17981790435075248
For : personal ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  2.1min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=300)
              precision    recall  f1-score   support

           0       0.96      0.88      0.92      2267
           1       0.97      0.99      0.98      9635

    accuracy                           0.97     11902
   macro avg       0.97      0.94      0.95     11902
weighted avg       0.97      0.97      0.97     11902

0.09369740456716343
For : promotions ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  2.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=400)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      9524
           1       0.92      0.84      0.88      2378

    accuracy                           0.95     11902
   macro avg       0.94      0.91      0.93     11902
weighted avg       0.95      0.95      0.95     11902

0.135284585012487
For : forums ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  1.8min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=200)
              precision    recall  f1-score   support

           0       0.97      0.98      0.98     10048
           1       0.90      0.86      0.88      1854

    accuracy                           0.96     11902
   macro avg       0.94      0.92      0.93     11902
weighted avg       0.96      0.96      0.96     11902

0.11152014929760716
For : purchases ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  1.1min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=300)
              precision    recall  f1-score   support

           0       0.99      1.00      1.00     11803
           1       0.97      0.39      0.56        99

    accuracy                           0.99     11902
   macro avg       0.98      0.70      0.78     11902
weighted avg       0.99      0.99      0.99     11902

0.01538688514033661
For : travel ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:   47.6s finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=400)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11872
           1       1.00      0.47      0.64        30

    accuracy                           1.00     11902
   macro avg       1.00      0.73      0.82     11902
weighted avg       1.00      1.00      1.00     11902

0.008765039096290275
For : spam ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:   47.8s finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=200)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11856
           1       0.94      0.72      0.81        46

    accuracy                           1.00     11902
   macro avg       0.97      0.86      0.91     11902
weighted avg       1.00      1.00      1.00     11902

0.0049807708961446566
For : social ...
Fitting 3 folds for each of 4 candidates, totalling 12 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  12 out of  12 | elapsed:  1.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=300)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     10700
           1       0.99      0.97      0.98      1202

    accuracy                           1.00     11902
   macro avg       0.99      0.98      0.99     11902
weighted avg       1.00      1.00      1.00     11902

0.028454086950584913

 
Mean loss =  0.07223835316392081
```

For {'n_estimators': range(200,601,100), 'criterion': ['gini','entropy']} RandomForest fore training 70% decimation 7

```
For : updates ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed: 13.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=600)
              precision    recall  f1-score   support

           0       0.95      0.96      0.95      7589
           1       0.92      0.91      0.92      4313

    accuracy                           0.94     11902
   macro avg       0.94      0.93      0.93     11902
weighted avg       0.94      0.94      0.94     11902

0.19063924532667043
For : personal ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed: 10.6min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=200)
              precision    recall  f1-score   support

           0       0.97      0.87      0.92      2267
           1       0.97      0.99      0.98      9635

    accuracy                           0.97     11902
   macro avg       0.97      0.93      0.95     11902
weighted avg       0.97      0.97      0.97     11902

0.10306722451685864
For : promotions ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed: 11.7min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=500)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      9524
           1       0.92      0.83      0.87      2378

    accuracy                           0.95     11902
   macro avg       0.94      0.91      0.92     11902
weighted avg       0.95      0.95      0.95     11902

0.13970123131519321
For : forums ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed:  8.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=400)
              precision    recall  f1-score   support

           0       0.97      0.98      0.98     10048
           1       0.91      0.85      0.88      1854

    accuracy                           0.96     11902
   macro avg       0.94      0.92      0.93     11902
weighted avg       0.96      0.96      0.96     11902

0.10324567540292685
For : purchases ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed:  5.0min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=400)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11803
           1       0.96      0.51      0.66        99

    accuracy                           1.00     11902
   macro avg       0.98      0.75      0.83     11902
weighted avg       1.00      1.00      1.00     11902

0.012285756187235219
For : travel ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed:  3.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=500)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11872
           1       1.00      0.37      0.54        30

    accuracy                           1.00     11902
   macro avg       1.00      0.68      0.77     11902
weighted avg       1.00      1.00      1.00     11902

0.006938616931694726
For : spam ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed:  3.7min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=400)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     11856
           1       0.96      0.54      0.69        46

    accuracy                           1.00     11902
   macro avg       0.98      0.77      0.85     11902
weighted avg       1.00      1.00      1.00     11902

0.0048510787720220305
For : social ...
Fitting 3 folds for each of 10 candidates, totalling 30 fits
[Parallel(n_jobs=3)]: Using backend LokyBackend with 3 concurrent workers.
[Parallel(n_jobs=3)]: Done  30 out of  30 | elapsed:  6.7min finished
Best classifier :  RandomForestClassifier(n_estimators=300)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     10700
           1       0.97      0.97      0.97      1202

    accuracy                           0.99     11902
   macro avg       0.99      0.98      0.98     11902
weighted avg       0.99      0.99      0.99     11902

0.03339820194500439

 
Mean loss =  0.07426587879970069
```



For {'n_estimators': range(60,221,20), 'criterion': ['gini','entropy']} RandomForest

```
For : updates ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  6.7min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  9.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=120)
              precision    recall  f1-score   support

           0       0.96      0.96      0.96      2530
           1       0.93      0.93      0.93      1438

    accuracy                           0.95      3968
   macro avg       0.94      0.94      0.94      3968
weighted avg       0.95      0.95      0.95      3968

0.16318548038712852
For : personal ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  4.5min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  6.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=180)
              precision    recall  f1-score   support

           0       0.97      0.90      0.93       755
           1       0.98      0.99      0.98      3213

    accuracy                           0.98      3968
   macro avg       0.97      0.95      0.96      3968
weighted avg       0.97      0.98      0.97      3968

0.10692021234645509
For : promotions ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  5.0min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  7.0min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=160)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      3175
           1       0.92      0.84      0.88       793

    accuracy                           0.95      3968
   macro avg       0.94      0.91      0.93      3968
weighted avg       0.95      0.95      0.95      3968

0.13045760886868846
For : forums ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  3.7min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  5.2min finished
Best classifier :  RandomForestClassifier(n_estimators=180)
              precision    recall  f1-score   support

           0       0.97      0.98      0.98      3350
           1       0.90      0.83      0.87       618

    accuracy                           0.96      3968
   macro avg       0.94      0.91      0.92      3968
weighted avg       0.96      0.96      0.96      3968

0.10246425988878213
For : purchases ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.3min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  3.2min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=60)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3935
           1       1.00      0.55      0.71        33

    accuracy                           1.00      3968
   macro avg       1.00      0.77      0.85      3968
weighted avg       1.00      1.00      1.00      3968

0.010270800498888547
For : travel ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  1.6min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  2.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=180)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3958
           1       1.00      0.30      0.46        10

    accuracy                           1.00      3968
   macro avg       1.00      0.65      0.73      3968
weighted avg       1.00      1.00      1.00      3968

0.030411964636344303
For : spam ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  2.1min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  2.9min finished
Best classifier :  RandomForestClassifier(criterion='entropy')
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3953
           1       1.00      0.60      0.75        15

    accuracy                           1.00      3968
   macro avg       1.00      0.80      0.87      3968
weighted avg       1.00      1.00      1.00      3968

0.004094619413536134
For : social ...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  3.7min
[Parallel(n_jobs=4)]: Done  54 out of  54 | elapsed:  5.1min finished
Best classifier :  RandomForestClassifier(n_estimators=180)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3567
           1       0.98      0.98      0.98       401

    accuracy                           1.00      3968
   macro avg       0.99      0.99      0.99      3968
weighted avg       1.00      1.00      1.00      3968

0.027231896518478193

 
Mean loss =  0.07187960531978767
```





For {'n_estimators': range(60,181,20), 'criterion': ['gini','entropy']} RandomForest

```
For : updates ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  6.4min finished
Best classifier :  RandomForestClassifier(n_estimators=160)
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      2530
           1       0.92      0.92      0.92      1438

    accuracy                           0.94      3968
   macro avg       0.93      0.93      0.93      3968
weighted avg       0.94      0.94      0.94      3968

0.1897930723377197
For : personal ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  6.5min finished
Best classifier :  RandomForestClassifier(n_estimators=160)
              precision    recall  f1-score   support

           0       0.92      0.76      0.83       755
           1       0.95      0.98      0.96      3213

    accuracy                           0.94      3968
   macro avg       0.93      0.87      0.90      3968
weighted avg       0.94      0.94      0.94      3968

0.17890223661079713
For : promotions ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  5.9min finished
Best classifier :  RandomForestClassifier(criterion='entropy')
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      3175
           1       0.90      0.82      0.86       793

    accuracy                           0.95      3968
   macro avg       0.93      0.90      0.91      3968
weighted avg       0.95      0.95      0.95      3968

0.14257311628782576
For : forums ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  4.0min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=140)
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      3350
           1       0.88      0.82      0.85       618

    accuracy                           0.96      3968
   macro avg       0.93      0.90      0.91      3968
weighted avg       0.95      0.96      0.95      3968

0.11761661368092312
For : purchases ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  2.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy')
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3935
           1       0.91      0.64      0.75        33

    accuracy                           1.00      3968
   macro avg       0.96      0.82      0.87      3968
weighted avg       1.00      1.00      1.00      3968

0.011502956051208617
For : travel ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  1.8min finished
Best classifier :  RandomForestClassifier(n_estimators=60)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3958
           1       1.00      0.50      0.67        10

    accuracy                           1.00      3968
   macro avg       1.00      0.75      0.83      3968
weighted avg       1.00      1.00      1.00      3968

0.029271457690782373
For : spam ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  2.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=60)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3953
           1       1.00      0.27      0.42        15

    accuracy                           1.00      3968
   macro avg       1.00      0.63      0.71      3968
weighted avg       1.00      1.00      1.00      3968

0.015981267923671928
For : social ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  3.2min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=180)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3567
           1       0.99      0.97      0.98       401

    accuracy                           1.00      3968
   macro avg       0.99      0.98      0.99      3968
weighted avg       1.00      1.00      1.00      3968

0.02562097964710709
Mean loss =  0.08890771252875447
```

For {'n_estimators': range(60,181,20), 'criterion': ['gini','entropy']} Random Forest

```
For : updates ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  7.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy')
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      2530
           1       0.91      0.91      0.91      1438

    accuracy                           0.94      3968
   macro avg       0.93      0.93      0.93      3968
weighted avg       0.94      0.94      0.94      3968

0.18570867085153012
For : personal ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  8.8min finished
Best classifier :  RandomForestClassifier(n_estimators=140)
              precision    recall  f1-score   support

           0       0.90      0.71      0.79       755
           1       0.94      0.98      0.96      3213

    accuracy                           0.93      3968
   macro avg       0.92      0.85      0.87      3968
weighted avg       0.93      0.93      0.93      3968

0.19106941813188133
For : promotions ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  6.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=140)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      3175
           1       0.90      0.82      0.86       793

    accuracy                           0.95      3968
   macro avg       0.93      0.90      0.91      3968
weighted avg       0.95      0.95      0.95      3968

0.1477553477229934
For : forums ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  5.3min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=180)
              precision    recall  f1-score   support

           0       0.96      0.98      0.97      3350
           1       0.88      0.78      0.83       618

    accuracy                           0.95      3968
   macro avg       0.92      0.88      0.90      3968
weighted avg       0.95      0.95      0.95      3968

0.12047922222080612
For : purchases ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  3.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=180)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3935
           1       0.91      0.61      0.73        33

    accuracy                           1.00      3968
   macro avg       0.95      0.80      0.86      3968
weighted avg       1.00      1.00      1.00      3968

0.01192683099340546
For : travel ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  2.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy')
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3958
           1       1.00      0.60      0.75        10

    accuracy                           1.00      3968
   macro avg       1.00      0.80      0.87      3968
weighted avg       1.00      1.00      1.00      3968

0.020371094971971395
For : spam ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  3.5min finished
Best classifier :  RandomForestClassifier(n_estimators=80)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3953
           1       0.75      0.40      0.52        15

    accuracy                           1.00      3968
   macro avg       0.87      0.70      0.76      3968
weighted avg       1.00      1.00      1.00      3968

0.007167044406912431
For : social ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  4.4min finished
Best classifier :  RandomForestClassifier()
              precision    recall  f1-score   support

           0       0.99      1.00      1.00      3567
           1       0.99      0.96      0.97       401

    accuracy                           0.99      3968
   macro avg       0.99      0.98      0.98      3968
weighted avg       0.99      0.99      0.99      3968

0.02892314423269916
```

For {'n_estimators': range(40,161,20), 'criterion': ['gini','entropy']} Random forest : 

```
For : updates ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  4.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=160)
              precision    recall  f1-score   support

           0       0.95      0.95      0.95      2530
           1       0.91      0.91      0.91      1438

    accuracy                           0.94      3968
   macro avg       0.93      0.93      0.93      3968
weighted avg       0.94      0.94      0.94      3968

0.21481272952763228
For : personal ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  4.5min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=140)
              precision    recall  f1-score   support

           0       0.88      0.72      0.79       755
           1       0.94      0.98      0.96      3213

    accuracy                           0.93      3968
   macro avg       0.91      0.85      0.87      3968
weighted avg       0.93      0.93      0.92      3968

0.18387673032337917
For : promotions ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  5.2min finished
Best classifier :  RandomForestClassifier(n_estimators=160)
              precision    recall  f1-score   support

           0       0.95      0.98      0.97      3175
           1       0.91      0.81      0.86       793

    accuracy                           0.95      3968
   macro avg       0.93      0.89      0.91      3968
weighted avg       0.94      0.95      0.94      3968

0.16699515160967404
For : forums ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  3.8min finished
Best classifier :  RandomForestClassifier()
              precision    recall  f1-score   support

           0       0.97      0.98      0.97      3350
           1       0.87      0.81      0.84       618

    accuracy                           0.95      3968
   macro avg       0.92      0.89      0.91      3968
weighted avg       0.95      0.95      0.95      3968

0.12095347298739638
For : purchases ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  1.8min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=160)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3935
           1       0.96      0.70      0.81        33

    accuracy                           1.00      3968
   macro avg       0.98      0.85      0.90      3968
weighted avg       1.00      1.00      1.00      3968

0.01820635392047978
For : travel ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  1.4min finished
Best classifier :  RandomForestClassifier(criterion='entropy', n_estimators=160)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3958
           1       1.00      0.70      0.82        10

    accuracy                           1.00      3968
   macro avg       1.00      0.85      0.91      3968
weighted avg       1.00      1.00      1.00      3968

0.004224462419553143
For : spam ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  1.8min finished
Best classifier :  RandomForestClassifier(n_estimators=120)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3953
           1       0.73      0.53      0.62        15

    accuracy                           1.00      3968
   macro avg       0.86      0.77      0.81      3968
weighted avg       1.00      1.00      1.00      3968

0.008975658268579413
For : social ...
Fitting 3 folds for each of 14 candidates, totalling 42 fits
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done  42 out of  42 | elapsed:  2.5min finished
Best classifier :  RandomForestClassifier(n_estimators=160)
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      3567
           1       0.98      0.98      0.98       401

    accuracy                           1.00      3968
   macro avg       0.99      0.99      0.99      3968
weighted avg       1.00      1.00      1.00      3968

0.02632385811709799
```



For Ada depth= 5	 		n=90 

```
0.5582314297181173
0.5595542656974936
0.5032536402981271
0.4538640549236412
0.045209956846011645
0.04819364394774019
0.04490882352394458
0.16035688396094022
0.296696587364502
```



For Ada depth= 5  			n=80 

```
0.5750317550344562
0.5727567725309054
0.48925967047628666
0.4110752478754118
0.04223451915559334
0.009993134812717255
0.088931584526305
0.14358042970189708

0.29160788926419656
```



For Ada depth = 3 			n =300

```
0.6562861778232164
0.6557859390125982
0.6319553957272638
0.62855966915162
0.357122790123184
0.19022503307282349
0.3509695060151424
0.5053775604258937

0.4970352589189677
```