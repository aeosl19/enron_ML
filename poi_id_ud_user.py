#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

r = 42

data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

feature_list = ['poi',
               'bonus',
               'salary',
               'deferral_payments',
               'deferred_income',
               'director_fees',
               'exercised_stock_options',
               'expenses',
               'total_payments',
               'total_stock_value',
               'from_messages',
               'from_poi_to_this_person',
               'from_this_person_to_poi',
               'loan_advances',
               'long_term_incentive',
               'other',
               'restricted_stock',
               'restricted_stock_deferred',
               'salary',
               'shared_receipt_with_poi',
               'to_messages'
               ]

data = featureFormat(data_dict, feature_list)


import pprint
pp = pprint.PrettyPrinter(depth=6)

import copy
my_dataset = copy.deepcopy(data_dict)
my_feature_list = copy.deepcopy(feature_list)

for k in my_dataset.keys():
    my_dataset[k]['ratio_to_poi_to_all_sent']  = 0
    if (my_dataset[k]['from_poi_to_this_person'] != 'NaN') and (my_dataset[k]['from_messages'] != 'NaN') and (my_dataset[k]['from_messages'] != 0):
        my_dataset[k]['ratio_to_poi_to_all_sent'] = float(my_dataset[k]['from_this_person_to_poi'])/float(my_dataset[k]['from_messages'])

    my_dataset[k]['ratio_from_poi_to_all_received']  = 0
    if (my_dataset[k]['from_this_person_to_poi'] != 'NaN') and (my_dataset[k]['to_messages'] != 'NaN') and (my_dataset[k]['to_messages'] != 0):
        my_dataset[k]['ratio_from_poi_to_all_received'] = float(my_dataset[k]['from_poi_to_this_person'])/float(my_dataset[k]['to_messages'])


for i in ['ratio_to_poi_to_all_sent','ratio_from_poi_to_all_received']:
    print "II en:",i
    if i not in my_feature_list:
        my_feature_list.append(i)


## DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.cross_validation import StratifiedShuffleSplit
my_data = featureFormat(my_dataset, my_feature_list, sort_keys = True)
labels, feature_values = targetFeatureSplit(my_data)
folds = 1000
cv = StratifiedShuffleSplit(
     labels, folds, random_state=r)

clf = DecisionTreeClassifier(random_state=r)
steps = [
    ('scale', MinMaxScaler()),
    ('select_features',SelectKBest(f_classif)),
    ('my_classifier', clf)
    ]

parameters = dict(select_features__k=[3,5,9,15,19,21,'all'])

pipe = Pipeline(steps)

grid = GridSearchCV(pipe, param_grid=parameters, cv=cv, verbose=1, scoring='f1', n_jobs=4)

grid.fit(feature_values, labels)

print("The best parameters are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))

## shows F1 score of 0.3086  - The best parameters are {'select_features__k': 19} with a score of 0.3086


from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline, FeatureUnion


from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics.classification import classification_report

dt_pipeline_steps = [
    ('scale', MinMaxScaler()),
    ('select_features',SelectKBest(f_classif,k=19)),
    ('my_classifier', DecisionTreeClassifier(random_state=r))
    ]

dt_classifier = Pipeline(dt_pipeline_steps)
# dt_classifier.predict(feature_values)
#
# print classification_report(labels, feature_values)

test_classifier(dt_classifier, my_dataset, my_feature_list)


## shows F1 score of 0.33659  - F1: 0.33659