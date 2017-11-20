# python quiz
# required model decision tree, naive bayes, SVM
# extra: I'll also try using randomforest, adaboost, gradient boost and xgboost


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import sklearn.metrics as m


train = pd.read_csv('credit_clients.csv', header=1)

train.columns

# get the first row ID and delete it from the training set

train_id = train['ID']
train_y = train['default payment next month']

train = train.drop(['ID', 'default payment next month'], axis=1)

train.shape  # (30000, 23) 23 features included


# now let's try apply some algorithms on it and evaluate
# let's make a function to do it

classifiers = {'Gradient Boosting': GradientBoostingClassifier(),
               'Adaptive Boosting': AdaBoostClassifier(),
               'SVM': SVC(),
               'Random Forest': RandomForestClassifier(n_estimators=300),
               'Decision Tree': DecisionTreeClassifier(),
               'Naive Bayes': GaussianNB(),
               'XGBoost': xgb.XGBClassifier(max_depth=3, n_estimators=300,
                                            learning_rate=0.05)}

# evaluation is accuracy, F1, MCC, G-means

# since there is no g-mean measure in general modules
# I need to create a function for it


def g_mean(y_actual, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred)
    gmean_scores = np.sqrt((tn/float(tn+fp)) * (tp/(tp+fn)))
    return gmean_scores


def model_scores(train, train_y, classifiers):
    # assign new variables
    Classifier = []
    Accuracy = []
    MCC = []
    G_means = []
    F1 = []
    # to apply the evaluation functions in cross-validation
    # need to transform the score into a scorer
    m1 = m.make_scorer(m.accuracy_score)
    m2 = m.make_scorer(m.matthews_corrcoef)
    m3 = m.make_scorer(m.f1_score)
    m4 = m.make_scorer(g_mean)  # implement g_mean from the outside function
    for name, model in classifiers.items():
        print name
        # training model use 5-fold
        accuracy = cross_val_score(model, train, train_y, cv=5, scoring=m1)
        mcc = cross_val_score(model, train, train_y, cv=5, scoring=m2)
        f1 = cross_val_score(model, train, train_y, cv=5, scoring=m3)
        g_mean = cross_val_score(model, train, train_y, cv=5, scoring=m4)
        # calculate the mean
        Accuracy.append(np.mean(accuracy))
        MCC.append(np.mean(mcc))
        G_means.append(np.mean(g_mean))
        F1.append(np.mean(f1))
        Classifier.append(name)

    # store the result in a data frame
    res = pd.DataFrame({'classifier': Classifier,
                        'MCC': MCC,
                        'G_means': G_means,
                        'Accuracy': Accuracy,
                        'F1_measure': F1})
    return res


score_frame = model_scores(train, train_y, classifiers)

print score_frame

# really wait a long time! took over 10 minutes to calculate!!!
# here is the result

#    Accuracy  F1_measure   G_means       MCC         classifier
# 0  0.727267    0.404366  0.413955  0.226465      Decision Tree
# 1  0.821001    0.472831  0.363176  0.403882            XGBoost
# 2  0.379068    0.386582  0.884422  0.122684        Naive Bayes
# 3  0.820401    0.473113  0.364833  0.402459  Gradient Boosting
# 4  0.816834    0.437495  0.322638  0.379941  Adaptive Boosting
# 5  0.816034    0.472761  0.370860  0.390832      Random Forest
# 6  0.752598    0.423122  0.390342  0.256723                SVM

# so in conclusion, in terms of f-measure of decision tree,
# naive bayes and svm, svm did the best, slightly better than decison tree
# and naive bayes is the worst

# but in a whole scale, among all the algorithms I tried, the best f-measure
# goes to gradient boosting!

# so let's use gradient boosting to select the most important features

clf = GradientBoostingClassifier().fit(train, train_y)
importances = clf.feature_importances_

importance_frame = pd.DataFrame({'feature': train.columns,
                                 'importance': importances})

print importance_frame.sort_values(['importance'], ascending=False)

#       feature  importance
# 5       PAY_0    0.160693
# 11  BILL_AMT1    0.122392
# 0   LIMIT_BAL    0.061281
# 13  BILL_AMT3    0.057712
# 12  BILL_AMT2    0.054611
# 17   PAY_AMT1    0.044687
# 14  BILL_AMT4    0.042455
# 16  BILL_AMT6    0.041446
# 18   PAY_AMT2    0.037985
# 6       PAY_2    0.037480
# 4         AGE    0.037262
# 22   PAY_AMT6    0.033985
# 15  BILL_AMT5    0.033109
# 10      PAY_6    0.032585
# 7       PAY_3    0.027732
# 8       PAY_4    0.027438
# 21   PAY_AMT5    0.027267
# 3    MARRIAGE    0.026280
# 2   EDUCATION    0.025179
# 20   PAY_AMT4    0.021705
# 19   PAY_AMT3    0.020289
# 9       PAY_5    0.016926
# 1         SEX    0.009502

# now try use voting classifier to vote for the final result

eclf = VotingClassifier(estimators=[('naive_bayes', GaussianNB()),
                                    ('decision tree', DecisionTreeClassifier()),
                                    ('SVM', AdaBoostClassifier())], voting='hard')


accuracy_scores = cross_val_score(eclf, train, train_y, scoring='accuracy')
np.mean(accuracy_scores)
# accuracy
# 0.75076666666666669

f1_scores = cross_val_score(eclf, train, train_y, scoring='f1')
np.mean(f1_scores)

# f1
# 0.46232413321373916

mcc_scores = cross_val_score(eclf, train, train_y,
                             scoring=m.make_scorer(m.matthews_corrcoef))
np.mean(mcc_scores)

# mcc
# 0.29908171792541555

gmean_scores = cross_val_score(eclf, train, train_y,
                               scoring=m.make_scorer(g_mean))
np.mean(mcc_scores)

# g-mean
# 0.47694394213381558

# from the result, we can see all the values are slightly higher than
# the single classifier

# in addition, we can also explore the best parameters
# take Random Forest as an example

parameters = {'n_estimators': [100, 300, 500]}
rf = RandomForestClassifier()
clf = GridSearchCV(rf, parameters, scoring='f1')
clf.fit(train, train_y)

# check out the best parameters

clf.best_estimator_

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_split=1e-07, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             n_estimators=300, n_jobs=1, oob_score=False, random_state=None,
#             verbose=0, warm_start=False)

# looks like that 300 trees are the best

# and extra, I wanna try lightgbm, which is a very new classifier
# released by Microsoft and is said to be more powerful and
# faster than xgboost

# LightGBM params

lgb_params = {}
lgb_params['learning_rate'] = 0.01
lgb_params['n_estimators'] = 1250
lgb_params['max_bin'] = 10
lgb_params['subsample'] = 0.8
lgb_params['subsample_freq'] = 10
lgb_params['colsample_bytree'] = 0.8
lgb_params['min_child_samples'] = 500

lgb_model = LGBMClassifier(**lgb_params)

# check how f1 scores

lgb_f1_scores = cross_val_score(lgb_model, train, train_y, scoring='f1')

np.mean(lgb_f1_scores)

# f1 score is 0.47361385446793069
# well, as far the best, but not outstanding

# changed from github
