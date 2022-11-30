# Final Project
## Made by: Francisco Javier Hernández Martín
### 0. Introduction
This document is the report for the Final Term Project for the subject Open Source Software. In this project we've been asked to analyze a dataset containing information about brain tumours, seeking for the model that describes bestly the data and achieves the highest accuracy among them.
### 1. General test without hyperparameters
For the first approach, and to give ourselves a first idea of how the models will interact with the data, I tested all of them without any hyperparameters.

First of all, we load all the models we're going to use in an array for easiness:

```python
models = [
    ('Perceptron',sklearn.linear_model.Perceptron()), 
    ('Logistic Regression', sklearn.linear_model.LogisticRegression()), 
    ('Tree Classifier', sklearn.tree.DecisionTreeClassifier()),
    ('RF', sklearn.ensemble.RandomForestClassifier())
    ]
```

After that, I initialize the array in which I'm going to store the analysis results:

```python
results_list = []
```

Then, for each model that we added, I am going to apply the steps we've been practicing:
1. Training the model with 'fit' and the train sets
2. Predicting the data with 'predict' and X's test set
3. Measuring the accuracy of the prediction

```python
for name, model in models:
    clf = model.fit(X_train, y_train) # 1.
    y_pred = clf.predict(X_test) # 2.
    acc = sklearn.metrics.accuracy_score(y_test, y_pred) # 3.
    results_list.append((name, model, acc))
```

Lastly, I printed all the results analyzed in the format:

'Model name'

'--> Accuracy'

The data is sorted by the accuracy achieved by the model so we can get a ranking.

```python
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print("Results ordered by accuracy achieved by the model:")
for name, model, acc in results_list:
    print(name)
    print('  --> Accuracy: %.2f' % acc)
    print("--------------------")
```

And this is the result we got:

```r
Results ordered by accuracy achieved by the model:
RF
  --> Accuracy: 0.90
--------------------
Logistic Regression
  --> Accuracy: 0.81
--------------------
Tree Classifier
  --> Accuracy: 0.80
--------------------
Perceptron
  --> Accuracy: 0.78
--------------------
```

So, at first glance we can deduce that RandomForest is the best fitting model for the data. But I am going to go further and tune the hyperparameters on each model so we can see if there's any difference on the results.

## 2. Hyperparameter tuning
Now we are going to try to find the best parameters for each model to see if there's any change in the ranking we've previously obtained. To achieve that, I'm using GridSearchCV, in which we can pass the model and some of its parameters so it can try all the possible combinations of them and find the best of them (the one that outputs the highest accuracy). Some of the parameters are taken out of internet because they have already been tested by many users and it's been concluded that they are significant for the accuracy result, and some others have been added by me with my understanding of them and their relation to this dataset. This method takes time because of all the combinations that it has to try, but we can pass GridSearchCV the parameter 'n_jobs=-1' so it uses all the processors in your computer and it will reduce significantly the time spent in calculations.

### 2.1 RandomForest
For this model, the most important parameters are:
- 'max_depth': The maximum depth of the tree. Too much would cause overfitting, and too little would not explain correctly the data, this is why it's a critical parameter for this model.
- 'class_weight': The weight given to the different classes. None gives every class a weight of one.
- 'n_estimators': The number of trees in the forest. A large number of trees will increase the accuracy, and overfitting is reduced but it will make the model slower.
- 'min_samples_split': This is the minimum number of samples required to be at a leaf node.

```python
model=sklearn.ensemble.RandomForestClassifier()
param={'max_depth':[None,5,8],'class_weight':[None,'balanced'],'n_estimators':[100,200,300],'min_samples_split':[2,3,4]}

grid_model=sklearn.model_selection.GridSearchCV(model,param_grid=param, cv=5,scoring='f1_macro')
grid_model.fit(X_train, y_train)
rf_best_params = grid_model.best_params_
```

We discover that the best parameters, among those that we chose, are the following:

```r
{'class_weight': None,
 'max_depth': None,
 'min_samples_split': 4,
 'n_estimators': 200}
```

### 2.2 Logistic Regression
For this model, the most important parameters are:
- 'C': The C parameter controls the penality strength, which can also be effective.
- 'penalty': Regularization (penalty) can sometimes be helpful.
- 'solver': Different solving algorithms. Sometimes, you can see useful differences in performance or convergence with different solvers (solver).
```python
model=sklearn.linear_model.LogisticRegression()
param={'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],'penalty':['none','l1', 'l2','elasticnet'],'C':[100, 10, 1.0, 0.1, 0.01]}

grid_model=sklearn.model_selection.GridSearchCV(model,param_grid=param, cv=5,scoring='f1_macro')
grid_model.fit(X_train, y_train)
lr_best_params = grid_model.best_params_
```

We discover that the best parameters, among those that we chose, are the following:

```r
{'C': 10, 'penalty': 'l2', 'solver': 'newton-cg'}
```

### 2.3 Decission Tree
For this model, the most important parameters are:
- 'min_samples_leaf': Parameter used to control over-fitting by defining that each leaf has more than one element. In reality, what this is actually doing is simply just telling the tree that each leaf doesn’t have to have an impurity of 0, we will look into impurity further in min_impurity_decrease.
- 'max_depth': The maximum depth of the tree. Too much would cause overfitting, and too little would not explain correctly the data, this is why it's a critical parameter for this model.
```python
model = sklearn.tree.DecisionTreeClassifier()
param =  {
    'min_samples_leaf': [1, 2, 3],
    'max_depth': [3, 6, 10, None]
}

grid_model=sklearn.model_selection.GridSearchCV(model,param_grid=param, cv=5,scoring='f1_macro')
grid_model.fit(X_train, y_train)
dt_best_params = grid_model.best_params_
```

We discover that the best parameters, among those that we chose, are the following:

```r
{'max_depth': 10, 'min_samples_leaf': 1}
```

### 2.4 Perceptron
For this model, the most important parameters are:
- 'eta0': The perceptron's learning rate. A large learning rate can cause the model to learn fast, but perhaps at the cost of lower skill. A smaller learning rate can result in a better-performing model but may take a long time to train the model.
- 'max_iter': Another important hyperparameter is how many epochs are used to train the model.

```python
model = sklearn.linear_model.Perceptron()
param = {
    'eta0': [0.0001, 0.1, 1.0],
    'max_iter': [10, 1000, 10000],
}

grid_model=sklearn.model_selection.GridSearchCV(model,param_grid=param, cv=5,scoring='f1_macro')
grid_model.fit(X_train, y_train)
pc_best_params = grid_model.best_params_
```

We discover that the best parameters, among those that we chose, are the following:

```r
{'eta0': 1.0, 'max_iter': 1000}
```

### 2.5 Ranking the models with their optimal parameters
First of all, we load all the models, with the optimal parameters we have just obtained, in an array for easiness:

```python
models_with_params = [
    ('Perceptron',sklearn.linear_model.Perceptron(eta0=1.0, max_iter=1000)), 
    ('Logistic Regression', sklearn.linear_model.LogisticRegression(C=10, penalty='l2', solver='newton-cg')), 
    ('Tree Classifier', sklearn.tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=1)),
    ('RF', sklearn.ensemble.RandomForestClassifier(min_samples_split=4, n_estimators=200))
    ]
```

Now I initialize the new results list to add them there, and I proceed with the same procedure I used before to find the accuracy of the new models:
```python
results_list = []

for name, model in models_with_params:
    clf = model.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    results_list.append((name, model, acc))

results_list = sorted(results_list, key=lambda x: x[2], reverse=True)
print("Results ordered by accuracy achieved by the model:")
for name, model, acc in results_list:
    print(name)
    print('  --> Accuracy: %.2f' % acc)
    print("--------------------")
```

And we receive the following result:
```r
Results ordered by accuracy achieved by the model:
RF
  --> Accuracy: 0.90
--------------------
Logistic Regression
  --> Accuracy: 0.82
--------------------
Tree Classifier
  --> Accuracy: 0.80
--------------------
Perceptron
  --> Accuracy: 0.78
--------------------
```

And we notice that the result is pretty similar to the one obtained before, letting us know that RandomForest is the most optimal model for this data once again.

## 3. Most optimal model hyperparameters re-tuning
Now that we found that the RandomForest model is the most suitable for this dataset, I am going to try to tune its hyperparameters again to achieve the highest accuracy possible (even tho trying to achieve 100% is wrong, because that would mean overfitting).

Now we have to take in account the hyperparameters we used in the last step for RandomForest, and stop to think about them:
- 'min_samples_split': We obtained that 4 was the first best option, but we only tested number less than 4, so I'm going to give the model more options above 4 now (4, 6, 8).
- 'n_estimators': We receives that 200 was the first best option, and we tested 100, 200 and 300; so now I'm going to close the range to 150 and 200 only.
- 'max_features': New parameter to take into account for RandomForest model. This is the number of features to consider when looking for the best split. A Random Forest model can only have a maximum number of features in an individual tree. Many would assume that if you increase max_features, this will improve the overall performance of your model. However, this naturally decreases the diversity of individual trees which would also increase the time it took the model to produce outputs.
- 'min_samples_leaf': This is the minimum number of samples required to be at a leaf node. A leaf node is the end node of a decision tree and a smaller min_sample_leaf value will make the model more vulnerable to detecting noise.
- 'bootstrap': If this is set as False, the whole dataset is used to build each tree, but it is set as Default.
  
So now we have chosen the parameters to test, let's do it:
```python
model=sklearn.ensemble.RandomForestClassifier()
param={'n_estimators':[150,200],'min_samples_split':[4, 6, 8], 'max_features': ['auto', 'sqrt'], 'min_samples_leaf': [2, 4], 'bootstrap': [True, False]}

grid_model=sklearn.model_selection.GridSearchCV(model,param_grid=param, cv=5,scoring='f1_macro', n_jobs=-1)
grid_model.fit(X_train, y_train)
rf_best_params2 = grid_model.best_params_
```

This gives us the following supposedly best parameters:
```r
{'bootstrap': False,
 'max_features': 'auto',
 'min_samples_leaf': 2,
 'min_samples_split': 6,
 'n_estimators': 200}
```

And now we're going to test the accuracy score again, applying the model:
```python
best_model = sklearn.ensemble.RandomForestClassifier(bootstrap=False, max_features='sqrt', min_samples_leaf=2, min_samples_split=6, n_estimators=200)
clf = best_model.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

And this gives us an accuracy of **0.91**, the highest value we've obtained in this test:

```r
Accuracy: 0.91
```