#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.model_selection import train_test_split

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary',
                 'bonus',
                 'total_payments',
                 'exercised_stock_options',
                 'total_poi_emails',
                 'long_term_incentive',
                 'total_stock_value',
                 'restricted_stock'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### remove known outliers we know by name

data_dict.pop('TOTAL', None)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)

print data_dict['SKILLING JEFFREY K']

## check for NAN in value
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

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


## add scaler - standardize features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics.classification import classification_report


svc = SVC()
dTree = DecisionTreeClassifier(random_state=42)
rForest = RandomForestClassifier()
gnb = GaussianNB()
kBest = SelectKBest(f_classif)

### with DecisionTreeClassifier

steps = [('scaling',scaler), ('SKB', kBest), ('GNB', gnb)]
parameters = dict(SKB__k = [1,2,3,4,5,6,7,'all'])

pipe = Pipeline(steps)
grid = GridSearchCV(pipe, param_grid= parameters, scoring='f1')
grid.fit(features_train, labels_train)
clf = grid.best_estimator_
pred = clf.predict(features_test)


## print results
target_names = ['Non POI', 'POI']
report = classification_report(labels_test, pred, target_names= target_names)
print report
# print 'Presicion:', precision_score(labels_test, pred), 'recall:', recall_score(labels_test, pred)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

selected_features =  features_list[2], features_list[6]
#print selected_features
# print selected_features
dump_classifier_and_data(clf, my_dataset, features_list)
test_classifier(clf, my_dataset, features_list)
