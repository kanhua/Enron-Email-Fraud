#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

import pandas as pd
import numpy as np
import copy
from preprocess_data import FeatureSel,add_features
from sklearn.svm import LinearSVC
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features

features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees',
                 'exercised_stock_options', 'expenses', 'from_messages',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'long_term_incentive', 'other', 'restricted_stock',
                 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi',
                 'to_messages', 'total_payments', 'total_stock_value']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("./data/final_project_dataset.pkl", "r"))


### Task 2: Remove outliers

del data_dict["TOTAL"]
del data_dict["LOCKHART EUGENE E"]


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


my_dataset, new_feature_names, financial_features = add_features(data_dict)

features_list.extend(new_feature_names)

for f in financial_features:
    del features_list[features_list.index(f)]



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from preprocess_data import linearsvc_outlier_rm

# features,labels,_=linearsvc_outlier_rm(np.array(features),np.array(labels))

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


sd = StandardScaler()
fsl = FeatureSel(k_best=5, pca_comp=5)
# clf=Pipeline([("fsl",fsl),("sd",sd),("lvc",LinearSVC(C=0.000001))])


clf = Pipeline([("fsl", fsl), ("sd", sd), ("lvc", LinearSVC())])

gscv=GridSearchCV(clf,{"lvc__C":np.logspace(-6,-1,5),
                       "fsl__k_best":[1,5,10],
                       "fsl__pca_comp":[0,5,10]},
                  scoring="recall",verbose=0)


gscv.fit(np.array(features),np.array(labels))

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


test_classifier(gscv.best_estimator_, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(gscv.best_estimator_, my_dataset, features_list)
