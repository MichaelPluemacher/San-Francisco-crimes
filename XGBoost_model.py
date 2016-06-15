'''

Predicting crimes in San Francisco

A competition hosted by kaggle:
https://www.kaggle.com/c/sf-crime

Building an XGBoost classifier to
predict crimes by category

'''


import sys
from operator import itemgetter

import numpy
import pandas

import graphviz
import matplotlib.pyplot

from sklearn.feature_extraction import DictVectorizer

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn import cross_validation

# the XGBoost classifier

from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree

# set testingModels to True when testing models
# and to False for the final submission
testingModels = False

# load the cleaned and engineered data
train=pandas.read_csv('train_new.csv')
test=pandas.read_csv('test_new.csv')

# standard-scale the raw x-y

xy_scaler = preprocessing.StandardScaler() 
xy_scaler.fit(train[["X","Y"]]) 
train[["X","Y"]]=xy_scaler.transform(train[["X","Y"]]) 
test[["X","Y"]]=xy_scaler.transform(test[["X","Y"]])

# add some new features based on latitude and longitude:
# rotate coordinates by 30, 45 and 60 degreees
# and compute radial distance from center

train["rot45_X"] = 0.707* train["Y"] + 0.707* train["X"]
train["rot45_Y"] = 0.707* train["Y"] - 0.707* train["X"]
train["rot30_X"] = 0.866* train["X"] + 0.5* train["Y"]
train["rot30_Y"] = 0.866* train["Y"] - 0.5* train["X"]
train["rot60_X"] = 0.5* train["X"] + 0.866* train["Y"]
train["rot60_Y"] = 0.5* train["Y"] - 0.866* train["X"]
train["radial_r"] = numpy.sqrt( numpy.power(train["Y"],2) + numpy.power(train["X"],2) )

test["rot45_X"] = 0.707* test["Y"] + 0.707* test["X"]
test["rot45_Y"] = 0.707* test["Y"] - 0.707* test["X"]
test["rot30_X"] = 0.866* test["X"] + 0.5* test["Y"]
test["rot30_Y"] = 0.866* test["Y"] - 0.5* test["X"]
test["rot60_X"] = 0.5* test["X"] + 0.866* test["Y"]
test["rot60_Y"] = 0.5* test["Y"] - 0.866* test["X"]
test["radial_r"] = numpy.sqrt( numpy.power(test["Y"],2) + numpy.power(test["X"],2) )

################
#
# PCA and
# dimesional reduction
#
# slightly worsens the results
#
################

# trainNum = train.select_dtypes(include=['float64'])
# trainNonNum = train.select_dtypes(exclude=['float64'])
# testNum = test.select_dtypes(include=['float64'])
# testNonNum = test.select_dtypes(exclude=['float64'])
#
# trainNum.info()
# trainNonNum.info()

#
# print('Correlations between numerical variables in train')
# numpy.set_printoptions(threshold=numpy.nan)
# sys.stdout = open("Correlations.txt", "w")
# c, p = corrcoeff(trainNum, 'pearson')
# print(c)
# print('')
# print('corresponding p values')
# print(p)
# print('')
#
# sys.stdout = sys.__stdout__
#
# size = len(trainNum.columns)
#
# for i in range(size):
#     for j in range(i + 1, size):
#         if abs(c[i,j]) > 0.5:
#             print('i: ', i, ' j: ', j, ' correlation: ', c[i,j])
#
#
# # Principal component analysis
#
# print('Scale predictors...')
# scaler = preprocessing.StandardScaler().fit(trainNum)
#
# pca = PCA()
#
# print('Principal component analysis...')
# new_trainNum = pca.fit_transform(scaler.transform(trainNum))
#
# print(pca.explained_variance_ratio_)
#
# pca_n = 30
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# pca_n = 40
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# pca_n = 50
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# pca_n = 60
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# pca_n = 70
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# pca_n = 80
# print('Summing the ', pca_n, ' most important dimensions:')
# print(pca.explained_variance_ratio_[0:pca_n].sum())
#
# print('dimensionality reduction...')
# pca = PCA(n_components = 40)
#
# new_trainNum = pandas.DataFrame(pca.fit_transform(scaler.transform(trainNum)))
# new_testNum = pandas.DataFrame(pca.transform(scaler.transform(testNum)))
#
# #new_trainNum.info()
#
# # having reduced the number of dimensions,
# # we can reconstruct the data sets
# print('new training data:')
# train = trainNonNum.join(new_trainNum)
# train.info()
# print('new testing data:')
# test = testNonNum.join(new_testNum)
# test.info()

# for testing of models
#
# split into training and test set
# fix random_state to make the results reproducible
if testingModels:
   training, testing = cross_validation.train_test_split(train, test_size = 0.2, random_state=0)
   label = training['Category'].astype('category')
   testlabel = testing['Category'].astype('category')
   del training['Category']
   del testing['Category']
else:
   training = train
   testing = test

# small dataframes for debugging
#training = training[0:200000]
#testing = testing[0:200000]

label = training['Category'].astype('category')
testID = testing['Id']
del training['Category']
del testing['Id']

# Label encoder
preProc = preprocessing.LabelEncoder()
training['PdDistrict'] = preProc.fit_transform(training.PdDistrict)
testing['PdDistrict'] = preProc.transform(testing.PdDistrict)
training['DayOfWeek'] = preProc.fit_transform(training.DayOfWeek)
testing['DayOfWeek'] = preProc.transform(testing.DayOfWeek)

training.info()

# one-hot encoding
# training = training.T.to_dict().values()
# testing = testing.T.to_dict().values()
# vec = DictVectorizer(sparse = False)
# training = pandas.DataFrame(vec.fit_transform(training), columns=vec.get_feature_names())
# testing = pandas.DataFrame(vec.transform(testing), columns=vec.get_feature_names())
#
# training.info()

predictors = list(training.columns.values)
print(predictors)

# properly scale predictors
print('Scale predictors...')
scaler = preprocessing.StandardScaler().fit(training)
training = pandas.DataFrame(scaler.transform(training))
training = scaler.transform(training)
testing = scaler.transform(testing)

# stratified split of training data for testing models
stratShuffleSplit = cross_validation.StratifiedShuffleSplit(label, train_size = 0.5, n_iter = 1)

# perform grid search

xgb_model = XGBClassifier(n_estimators = 20,
                      learning_rate = 0.2,
                      max_depth = 11,
                      min_child_weight=4,
                      gamma = 0.4,
                      reg_alpha = 0.05,
                      reg_lambda = 2,
                      subsample = 1.0,
                      colsample_bytree = 1.0,
                      max_delta_step = 1,
                      scale_pos_weight = 1,
                      objective = 'multi:softprob',
                      nthread = 8,
                      seed = 0#,
                      #silent = False
)

# parameters to sample for xgb
#
# let's tune max_depth and
# min_child_weight first
#
# best model:
# max_depth = 9
# min_child_weight = 4
param_dist1 = {
               'max_depth':list(range(3,10,1)),
               'min_child_weight':list(range(1,6,1))
}
param_dist2 = {
    'max_depth': [9,11,13],
    'min_child_weight': [4,5]
}
# next we turn our attention to gamma
# best value gamma = 0.4
param_dist3 = {'gamma':[i/10.0 for i in range(0,5)]}
param_dist3 = {'gamma':[0.5,1,2]}

# now to subsample and colsample_bytree
# best results
# subsample = 1.0
# colsample_bytree = 1.0
param_dist4 = {
 'subsample':[0.4,0.6,0.8],
 'colsample_bytree':[0.4,0.6,0.8]
}
param_dist4 = {
 'subsample':[0.9,1.0],
 'colsample_bytree':[0.9,1.0]
}
# on to the regularization parameters
# for the tree booster
param_dist5 = {
  'reg_alpha':[1e-5, 1e-2, 0.1],
  'reg_lambda':[1, 3, 5]
}
param_dist5 = {
  'reg_alpha':[0.05, 0.2],
  'reg_lambda':[2, 4]
}

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              numpy.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

if testingModels:
    print('run grid search...')
    grid_search = GridSearchCV(xgb_model,
                                 param_grid = param_dist5,
                                 cv = stratShuffleSplit,
                                 scoring = 'log_loss',
                                 verbose=10)
    grid_search.fit(training, label)
    print('Report grid scores:')
    report(grid_search.grid_scores_)
    print('Best parameters')
    print(grid_search.best_params_)
    print('Best score')
    print(grid_search.best_score_)
else:
    xgb_model = XGBClassifier(n_estimators=20,
                              learning_rate=0.2,
                              max_depth=11,
                              min_child_weight=4,
                              gamma=0.4,
                              reg_alpha=0.05,
                              reg_lambda=2,
                              subsample=1.0,
                              colsample_bytree=1.0,
                              max_delta_step=1,
                              scale_pos_weight=1,
                              objective='multi:softprob',
                              nthread=8,
                              seed=0  # ,
                              # silent = False
                              )
    print('training...')
    xgb_model.fit(training, label)
    print('predicting...')
    predicted = xgb_model.predict_proba(testing)
    predicted = pandas.DataFrame(predicted)
    predicted.columns = xgb_model.classes_
    # Name index column.
    predicted.index.name = 'Id'
    # Write csv.
    print('Saving prediction...')
    predicted.to_csv('Prediction.csv')
    # feature importance
    feat_imp = pandas.Series(xgb_model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    matplotlib.pyplot.show()
    plot_importance(xgb_model, title='Feature importance')
    matplotlib.pyplot.show()
    plot_tree(xgb_model, num_trees=0)
    matplotlib.pyplot.show()
