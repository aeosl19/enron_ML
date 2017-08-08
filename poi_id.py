#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import matplotlib.pyplot as plt
import pandas as pd


features_list = ['poi',
                 'salary',
                 'bonus',
                 'restricted_stock',
                 'total_payments',
                 'total_stock_value',
                 'exercised_stock_options',
                 'from_this_person_to_poi',
                 'from_poi_to_this_person',
                 'long_term_incentive',
                 'director_fees',
                 'expenses',
                 'shared_receipt_with_poi',
                 'to_messages',
                 'from_messages',
                 'loan_advances',
                 'restricted_stock_deferred',
                 'bonus_salary_ratio',
                 'total_poi_emails'
                 ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## Features with NAN values
pois = 0
nans = dict()
for person in  data_dict:
    if data_dict[person]['poi'] == True:
        pois += 1

    for item in data_dict[person]:
        if data_dict[person][item] == 'NaN':
            if item in nans:
                nans[item] += 1
            else:
                nans[item] = 1

print "Number of Features with NaN values:", nans


## number of datapoints
total_datapoints = len(data_dict)
print "Total datapoints:", total_datapoints

## number of pois
print "Number of POIs:", pois
print "Number none POIs:", total_datapoints - pois

## remove items we know by name
data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

## check for NAN in value
def check_nan(data):
    if data == 'NaN':
        return 0.0
    return data


## Create new features total_poi_emails and bonus_salary_ratio

salaries = []
B_S_ratio = []

for person in data_dict:
    data_dict[person]['total_poi_emails'] = check_nan(data_dict[person]['from_this_person_to_poi']) +\
                                            check_nan(data_dict[person]['from_poi_to_this_person'])

    data_dict[person]['salary'] = check_nan(data_dict[person]['salary'])
    data_dict[person]['bonus'] = check_nan(data_dict[person]['bonus'])
    salaries.append(data_dict[person]['salary'])

    ## create bonus_salary_ratio feature
    if data_dict[person]['bonus'] == 0:
        data_dict[person]["bonus_salary_ratio"] = 0.
    elif data_dict[person]['salary'] == 0:
        data_dict[person]["bonus_salary_ratio"] = 1.
    else:
        data_dict[person]["bonus_salary_ratio"] = float(data_dict[person]['bonus']) / float(data_dict[person]['salary'])
    B_S_ratio.append(data_dict[person]["bonus_salary_ratio"])

# plt.scatter(salaries, B_S_ratio)
# plt.show()

## Person with a max bonus to salary ratio classified is a POI. Therefor I've choosen to not remove any outliers
print "Person with a max bonus to salary ratio:", data_dict.items()[B_S_ratio.index(max(B_S_ratio))]

my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### setup for tests###
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

cv = StratifiedShuffleSplit(labels, 1000, random_state= 42)

# ######### TEST 1 #############
# from sklearn.ensemble import AdaBoostClassifier
#
# steps = [('scaler', MinMaxScaler()), ('select', SelectKBest(f_classif)), ('pca', PCA()), ('ADA', AdaBoostClassifier())]
# pipe = Pipeline(steps)
# parameters = {'pca__whiten': [False],
#               'select__k' : [12],
#               'pca__n_components' : [6],
#               'ADA__n_estimators' : [70],
#               'ADA__learning_rate' : [1.5]
#                }
#
# grid = GridSearchCV(pipe,param_grid=parameters, cv= cv, n_jobs = -1, verbose=1, scoring='f1')
# grid.fit(features, labels)
# clf = grid.best_estimator_
# print "Estimator:", clf
# pred = clf.predict(features)
#
#
# ## get feature importance
# features_scores =  clf.named_steps['select'].scores_
# feature_selected =  clf.named_steps['select'].get_support()
#
# feature_score = []
# for feature, score, selected in zip(features_list[1:], features_scores, feature_selected):
#     feature_score.append([feature,score, selected])
#
# feature_score_df = pd.DataFrame(feature_score)
# print feature_score_df, '\n'
#
# ## Dump and test classifier ##
#
# # dump_classifier_and_data(clf, my_dataset, features_list)
# test_classifier(clf, my_dataset, features_list)
# print "\n END TEST ADA"



# ######### TEST 2 #############
#
# steps = [('scaler', MinMaxScaler()), ('select', SelectKBest(f_classif)), ('GNB', GaussianNB())]
# pipe = Pipeline(steps)
# parameters = {'select__k' : [4, 6, 8]}
#
# grid = GridSearchCV(pipe,param_grid=parameters, cv= cv, n_jobs = -1, verbose=1, scoring='f1')
# grid.fit(features,labels)
# clf = grid.best_estimator_
# print "Estimator:", clf
# selected_features =  clf.named_steps['select'].get_support()
# pred = clf.predict(features)
#
#
# # get features and importance
# features_scores =  clf.named_steps['select'].scores_
# features_pvalues = clf.named_steps['select'].pvalues_
# feature_selected =  clf.named_steps['select'].get_support()
#
# feature_score = []
#
# for feature, score, pvalue, selected in zip(features_list[1:], features_scores, features_pvalues, feature_selected):
#     feature_score.append([feature, score, pvalue, selected])
#
# feature_score_df = pd.DataFrame(feature_score)
# print feature_score_df
#
# ## Dump and test classifier ##
# # dump_classifier_and_data(clf, my_dataset, features_list)
# test_classifier(clf, my_dataset, features_list)
#
# print "\n END GNB TEST"


########## TEST 3 #############

from sklearn.svm import SVC

steps = [('scaler', MinMaxScaler()), ('select', SelectKBest(f_classif)), ('pca', PCA()), ('SVC', SVC())]
pipe = Pipeline(steps)
parameters = { 'select__k' : [14],
               'pca__n_components' : [2],
               'pca__whiten' : [False],
               'SVC__kernel' : ['linear'],
               'SVC__C' : [1, 2,4],
               'SVC__class_weight': [{True: .8, False: .1},
                                     ]
               }
grid = GridSearchCV(pipe,param_grid=parameters, cv= cv, n_jobs = -1, verbose=1, scoring='f1')
grid.fit(features, labels)
clf = grid.best_estimator_
print "Estimator:", clf.named_steps
pred = clf.predict(features)


# get features and importance
features_scores =  clf.named_steps['select'].scores_
features_pvalues = clf.named_steps['select'].pvalues_
feature_selected =  clf.named_steps['select'].get_support()

feature_score = []

for feature, score, pvalue, selected in zip(features_list[1:], features_scores, features_pvalues, feature_selected):
    feature_score.append([feature, score, pvalue, selected])

feature_score_df = pd.DataFrame(feature_score)
print feature_score_df

## Dump and test classifier ##
dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)


print "END SVC TEST"