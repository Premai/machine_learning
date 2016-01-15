#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi", "salary", "bonus", "fragment_from_poi_email", 
                 "fragment_to_poi_email","deferral_payments", "total_payments",
                 "loan_advances", "restricted_stock_deferred",
                 "deferred_income", "total_stock_value", "expenses", 
                 "exercised_stock_options","long_term_incentive", 
                 "shared_receipt_with_poi", "restricted_stock", 
                 "director_fees"]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('LOCKHART EUGENE E', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

### Task 3: Create new feature(s)
### Fragment of email from poi to a person and 
### fragment of email to poi from a person
def crea_nf_list(pkey,pmsg):
    nf_list=[]

    for i in data_dict:
        if data_dict[i][pkey]=="NaN" or data_dict[i][pmsg]=="NaN":
            nf_list.append(0.)
        elif data_dict[i][pkey]>=0:
            nf_list.append(float(data_dict[i][pkey])/float(data_dict[i][pmsg]))
    return nf_list
    
fragment_from_poi_email=crea_nf_list("from_poi_to_this_person","to_messages")
fragment_to_poi_email=crea_nf_list("from_this_person_to_poi","from_messages")

### insert the new features into data_dict
icount=0
for i in data_dict:
    data_dict[i]["fragment_from_poi_email"]=fragment_from_poi_email[icount]
    data_dict[i]["fragment_to_poi_email"]=fragment_to_poi_email[icount]
    icount +=1
    
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
### Split data into train and test sets
### Used SelectKBest to pick features based on their scores.
from sklearn.feature_selection import SelectKBest
k_best = SelectKBest(k=16)
k_best.fit(features, labels)
scores = k_best.scores_

unsorted_pairs = zip(features_list[1:], scores)
sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
k_best_features = dict(sorted_pairs[:16])

for i in range(16):
    print sorted_pairs[i]

print("\n")    

##Create the top picked features_list    
#features_list = ["poi", "fragment_from_poi_email", "fragment_to_poi_email",
#"shared_receipt_with_poi", "exercised_stock_options","total_stock_value",
#"bonus","salary"]
features_list = ["poi", "salary","bonus","shared_receipt_with_poi",
"total_stock_value", "fragment_to_poi_email","exercised_stock_options"]
data = featureFormat(my_dataset, features_list, sort_keys = True)

##Split data into labels and features (Since first feature in the array is 
##the label, POI would be the label)
labels, features = targetFeatureSplit(data)

## Cross Validators used for local testing

def cross_val_eval(clf): 
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=42)
    
    scaler = preprocessing.MinMaxScaler().fit(features_train)
    features_train_transformed = scaler.transform(features_train)
    clf = clf.fit(features_train_transformed, labels_train)
    features_test_transformed = scaler.transform(features_test)
    predicted= clf.predict(features_test_transformed)
    #clf = clf.fit(features_train, labels_train)
    #predicted= clf.predict(features_test)
    print "Validating algorithm using train_test_split (Local testing):"
    print 'precision = ', precision_score(labels_test,predicted)
    print 'recall = ', recall_score(labels_test,predicted)
    print 'accuracy = ', accuracy_score(labels_test, predicted)  

def cross_val_kfold(clf):
    ### use KFold for split and validate algorithm
    from sklearn.cross_validation import KFold
    from sklearn.feature_extraction.text import TfidfVectorizer
    kf=KFold(len(labels),3)
  #  print len(labels)
  #  print kf
    for train_indices, test_indices in kf:
        #make training and testing sets
        features_train= [features[ii] for ii in train_indices]
        features_test= [features[ii] for ii in test_indices]
        labels_train=[labels[ii] for ii in train_indices]
        labels_test=[labels[ii] for ii in test_indices]
        
    ## For local testing the below lines should be lined up with
    ## the above lines and not in line with for train_indices.  But since it 
    ## does 1000 iterations i have moved it for submission purpose.
    scaler = preprocessing.MinMaxScaler().fit(features_train)
    features_train_transformed = scaler.transform(features_train)
    clf = clf.fit(features_train_transformed,labels_train)
    features_test_transformed = scaler.transform(features_test)
    clf.fit(features_train_transformed, labels_train)
    predicted = clf.predict(features_test_transformed)
    print "\nValidating algorithm using kfold (Local testing):"
    print "accuracy tuning = ",accuracy_score (predicted,labels_test)
    print 'precision = ', precision_score(labels_test,predicted)
    print 'recall = ', recall_score(labels_test,predicted)
        
def cross_val_skfold(clf):
    from sklearn.cross_validation import StratifiedShuffleSplit
    ### use SKFold for split and validate algorithm
    cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
    ### fit the classifier using training set, and test on test set
    #clf.fit(features_train, labels_train)
    #predicted = clf.predict(features_test)
    ## For local testing the below lines should be lined up with
    ## the above lines and not in line with for train_idx.  But since it 
    ## does 1000 iterations i have moved it for submission purpose.
    scaler = preprocessing.MinMaxScaler().fit(features_train)
    features_train_transformed = scaler.transform(features_train)
    clf = clf.fit(features_train_transformed, labels_train)
    features_test_transformed = scaler.transform(features_test)
    predicted= clf.predict(features_test_transformed)
    print "\nValidating algorithm using skfold (Local testing):"
    print 'precision = ', precision_score(labels_test,predicted)
    print 'recall = ', recall_score(labels_test,predicted)
    print "accuracy tuning = ", accuracy_score(labels_test, predicted)
    

print 'Classifier evaluation before tuning.\n'

## GaussianNB
# Provided to give you a starting point. Try a varity of classifiers.
clf = GaussianNB() 

print 'Evaluating the performance of GaussianNB:\n'
cross_val_eval(clf)
cross_val_kfold(clf)
cross_val_skfold(clf)

print 'Classifier testing using test_classifier:\n'
test_classifier(clf, my_dataset, features_list)

clf = KNeighborsClassifier()

##KNeighbors
print 'Evaluating the performance of KNeighborsClassifier:\n'
cross_val_eval(clf)
cross_val_skfold(clf)
cross_val_kfold(clf)

print 'Classifier testing using test_classifier:\n'
test_classifier(clf, my_dataset, features_list)

## DecisionTree Classifier
clf = DecisionTreeClassifier()

print 'Evaluating the performance DecisionTreeClassifier:\n'
cross_val_eval(clf)
cross_val_skfold(clf)
cross_val_kfold(clf)

print 'Classifier testing using test_classifier:\n'
test_classifier(clf, my_dataset, features_list)

print("\n")
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print 'Classifier evaluation after tuning.\n'

parameters = {'n_neighbors':[5,6,7,8],'weights' : ('distance', 'uniform') }
svr = KNeighborsClassifier(algorithm = 'brute')
clf = GridSearchCV(svr, parameters,scoring = 'recall')

print 'Evaluating the performance of KNeighborsClassifier:\n'
cross_val_eval(clf)
cross_val_skfold(clf)
cross_val_kfold(clf)

print 'Evaluate classifier performance using test_classifier:\n'
test_classifier(clf, my_dataset, features_list)

clf = DecisionTreeClassifier(criterion='entropy',random_state=42)
#parameters = {'criterion':['entropy'],'random_state':[42]}
#clf = GridSearchCV(clf, parameters)

print 'Evaluating the performance DecisionTreeClassifier:\n'
cross_val_eval(clf)
cross_val_skfold(clf)
cross_val_kfold(clf)

print 'Evaluate classifier performance using test_classifier:\n'
test_classifier(clf, my_dataset, features_list)

print("\n")

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)