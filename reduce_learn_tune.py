# Import libraries necessary for this project
import numpy as np
import pandas as pd
#import visuals as vs # Supplementary code

import matplotlib.pyplot as plt

# %matplotlib inline


from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.grid_search import GridSearchCV


my_random_seed = 66

#
# Functions leveraged from Udacity Student Intervention project work
#
 
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    y_pred = clf.predict(features)
 
    # Print and return results
    return f1_score(target.values, y_pred, pos_label='yes', average='micro')

def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier

    clf.fit(X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))
    
    
# Load the dataset
data = pd.read_csv('data.csv',index_col=0, parse_dates=True)

print "== Load data ", "="*40  
print "The dataset has {} data points with {} variables each.".format(*data.shape)

print data.columns

print "== Remove constant or unneeded data  " 
data.drop(['[CPU]Nice%','[CPU]Irq%','[CPU]Steal%','[MEM]Tot', '[MEM]Shared', '[MEM]Locked', '[MEM]SwapTot', \
           '[MEM]Clean', '[MEM]Laundry', '[MEM]HugeTotal', '[MEM]HugeFree', '[MEM]HugeRsvd', '[NET]RxCmpTot', \
           '[NET]RxMltTot', '[NET]TxCmpTot', '[NET]RxErrsTot', '[NET]TxErrsTot', \
           '[CPU]L-Avg1', '[CPU]L-Avg5', '[CPU]L-Avg15' ], axis = 1, inplace=True )

print(data.shape)

labels =  data['workload']
features = data.drop(['workload','benchmark' ], axis = 1) 

print "* Feature shape:", features.shape
print "* Labels shape:", labels.shape


# Shuffle and split the dataset

X_train, X_test, y_train, y_test  = cross_validation.train_test_split(features, labels, stratify=labels, test_size=0.25, random_state=my_random_seed)

# Show the results of the split
print "* Training set has {} samples.".format(X_train.shape[0])
print "* Testing set has {} samples.".format(X_test.shape[0])

# print "* X Train mean {}".format(X_train.mean(axis=0))
# print "* X Train stddev:", X_train.std(axis=0)


print "== 1st basic training on original data", "-"*40

#  Initialize the three models
clf_A = DecisionTreeClassifier(random_state=my_random_seed)
clf_B = SVC(random_state=my_random_seed)
clf_C = LinearSVC(C=0.1)
clf_D = GaussianNB()


# loop thru models, then thru train sizes
for clf in [clf_A, clf_B, clf_C, clf_D]:
    print "\n{}: \n".format(clf.__class__.__name__)
    train_predict(clf, X_train, y_train, X_test, y_test)
    scores = cross_validation.cross_val_score(clf, features, labels, cv=5)
    print "cross_validation:", scores, scores.mean()

        
print "== Scale data " 

scaler = StandardScaler()
#scaled_features = scaler.fit_transform( features )

scaled_features = features
scaled_features.ix[:,0:] = scaler.fit_transform(scaled_features.ix[:,0:])

# Shuffle and split the dataset

scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test  = cross_validation.train_test_split(scaled_features, labels, stratify=labels, test_size=0.25, random_state=my_random_seed)

# Show the results of the split
print "* scaled_Training set has {} samples.".format(scaled_X_train.shape[0])
print "* scaled_Testing set has {} samples.".format(scaled_X_test.shape[0])

print "== 2nd basic training on scaled data", "-"*40

# loop thru models, then thru train sizes
for clf in [clf_A, clf_B, clf_C, clf_D]:
    print "\n{}: \n".format(clf.__class__.__name__)
    train_predict(clf, scaled_X_train, scaled_y_train, scaled_X_test, scaled_y_test)
    scores = cross_validation.cross_val_score(clf, scaled_features, labels, cv=5)
    print "cross_validation:", scores, scores.mean()


print "== Feature selection ", "="*40  

print "-- 4: RFE on scaled", "-"*40 
# feature extraction

smodel = LogisticRegression()
srfe = RFE(smodel, 10)
sfit = srfe.fit(scaled_features, labels)
print("Num Features: %d") % sfit.n_features_

for i,v in enumerate(sfit.support_):
   if v == True:
      print i,v, sfit.ranking_[i], features.columns[i]

sr = []
for i,v in enumerate(sfit.support_):
   if v == False:
      sr.append(i)

srfeatures = features.drop(features.columns[sr],axis=1)
print srfeatures.columns

# Shuffle and split the dataset

sr_X_train, sr_X_test, sr_y_train, sr_y_test  = cross_validation.train_test_split(srfeatures, labels, stratify=labels, test_size=0.25, random_state=my_random_seed)

print "\n== Training on scaled and reduced features ", "-"*40

# loop thru models
for clf in [clf_A, clf_B, clf_C, clf_D]:
    print "\n{}: \n".format(clf.__class__.__name__)
    train_predict(clf, sr_X_train, sr_y_train, sr_X_test, sr_y_test)
    scores = cross_validation.cross_val_score(clf, srfeatures, labels, cv=5)
    print "cross_validation:", scores, scores.mean()
    
    
print "\n== GridSearchCV ... ", "-"*40

print "\n-- Linear SVC ... ", "-"*40
Cs = [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = {'C': Cs}

print(param_grid)

grid_search = GridSearchCV(LinearSVC(), param_grid, verbose=1, cv=5)
grid_search.fit(srfeatures, labels)

print(grid_search.best_params_)
print(grid_search.best_score_)

print "\n-- SVC ... ", "-"*40

Cs = [0.001, 0.01, 0.1, 1, 10, 100]
gammas = [0.001, 0.01, 0.1, 1, 10]


param_grid = {'C': Cs, 'gamma' : gammas}

print(param_grid)

# GridSearchCV on the whole dataset 
grid_search = GridSearchCV(SVC(), param_grid, verbose=1, cv=5)
grid_search.fit(srfeatures, labels)

print(grid_search.best_params_)
print(grid_search.best_score_)

scores = [x[1] for x in grid_search.grid_scores_]
scores = np.array(scores).reshape(6, 5)

plt.matshow(scores)
plt.xlabel('gamma')
plt.ylabel('C')
plt.colorbar()
plt.xticks(np.arange(5), param_grid['gamma'])
plt.yticks(np.arange(6), param_grid['C']);

# To avoid overfitting, do GridSearchCV only on training data
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(sr_X_train, sr_y_train)

# print grid_search.predict(sr_X_test)

print "Score for training set: {:.4f}.".format(grid_search.score(sr_X_train, sr_y_train))
print "Score for test set: {:.4f}.".format(grid_search.score(sr_X_test, sr_y_test))
    
# Do Cross vaildation with all data   
scores =  cross_validation.cross_val_score(grid_search, srfeatures, labels, cv=5)
print "cross_validation:", scores, scores.mean()

