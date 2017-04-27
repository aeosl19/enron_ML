#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'exercised_stock_options',
                 'total_poi_emails', 'long_term_incentive', 'total_stock_value', 'restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### remove known outliers we know by name

data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

## check for NAN i value

def check_nan(data):
    if data == 'NaN':
        return 0.0
    return data


for person in data_dict:
    data_dict[person]['total_poi_emails'] = check_nan(data_dict[person]['from_this_person_to_poi']) +\
                                            check_nan(data_dict[person]['from_poi_to_this_person'])

# print data_dict['SKILLING JEFFREY K']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# import matplotlib.pyplot as plt
# bonus = []
# total_poi_emails = []
#
# for person in my_dataset:
#     bonus.append(check_nan(my_dataset[person]['bonus']))
#     total_poi_emails.append(my_dataset[person]['total_poi_emails'])
#
# plt.scatter(bonus, total_poi_emails)
# plt.show()


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()


## add scaler - standardize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.classification import classification_report

# rfc = RandomForestClassifier()
svc = SVC()
# steps = [("rfc_classifier", rfc)]
# parameters = {"rfc_classifier__min_samples_split" : [2, 3, 4]}
steps = [('scaling',scaler),("svc_classifier", svc)]
parameters = {'svc_classifier__kernel':('linear', 'rbf'), 'svc_classifier__C':[1, 10]}
pipe = Pipeline(steps)
# print "test:", pipe.get_params().keys()
clf = GridSearchCV(pipe, param_grid= parameters)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
report = classification_report(labels_test, pred)
print report


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
#test_classifier(clf, my_dataset, features_list)