from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from deep import NN
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

shape = (4)
n_classes = 3
predict = []
models = []
models.append(['LR', LogisticRegression()])
models.append(["LDA", LinearDiscriminantAnalysis()])
models.append(["KNN", KNeighborsClassifier()])
models.append(["CART", DecisionTreeClassifier()])
models.append(["NB", GaussianNB()])
models.append(["SVM", SVC()])
models.append(['XB', XGBClassifier()])
models.append(["RFC", RandomForestClassifier()])
models.append(["NN", NN(shape = (4), n_classes=3)])
params = [{"C": [0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100], 'penalty':['l1', 'l2', 'elasticnet', 'none'],
           "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
          {"solver": ["svd", "lsqr", "eigen"]},
          {'n_neighbors':[5,6,7,8,9,10], 'leaf_size':[1,2,3,5,10,15,20,25,30], 'weights':['uniform', 'distance'], 'algorithm':['auto', 'ball_tree','kd_tree','brute']},
          {"criterion": ['gini', 'entropy'], 'max_depth':[None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "min_samples_split": list(range(2, 16)),
           "min_samples_leaf": list(range(1, 11))},
          {},
          {"C":[0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100], "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
           "gamma": [0.001, 0.05, 0.01, 0.5, 0.1, 1, 5, 10, 50, 100]},
          {"booster": ['gbtree', 'gblinear', 'dart'], "learning_rate":[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
           'min_child_weight':[1,3,5],'gamma':[0,1,2,3],'colsample_bytree':[0.5,0.8, 1],"subsample":[0.5, 0.8, 1]},
          {'n_estimators': list(range(10, 101, 5)), 'criterion': ['gini', 'entropy'], 'min_samples_split': list(range(2, 10)),
           'min_samples_leaf': list(range(1, 6)), 'n_jobs': [-1]},
          {}]

best_param = []

def cross_val_boxplot(x, y, num_folds):
    results = []
    names = []
    for (name, model), p in zip(models, best_param):
        kfold = StratifiedKFold(n_splits=num_folds)
        if name == 'LR':
            model = model.set_params(C=p['C'], penalty=p['penalty'], solver = p['solver'])
        elif name == 'LDA':
            model = model.set_params(solver = p['solver'])
        elif name == 'KNN':
            model = model.set_params(n_neighbors = p['n_neighbors'], leaf_size = p['leaf_size'], weights = p['weights'], algorithm = p['algorithm'])
        elif name == 'CART':
            model = model.set_params(criterion = p['criterion'], max_depth = p['max_depth'], min_samples_split = p['min_samples_split'],
                                     min_samples_leaf = p['min_samples_leaf'])
        elif name == "SVM":
            model = model.set_params(C = p['C'], kernel = p['kernel'], gamma = p['gamma'], probability = True)
        elif name == "XB":
            model = model.set_params(booster = p['booster'], learning_rate = p['learning_rate'], min_child_weight = p['min_child_weight'],
                                     gamma = p['gamma'], colsample_bytree = p['colsample_bytree'], subsample = p['subsample'])
        elif name == "RFC":
            model = model.set_params(n_estimators = p['n_estimators'], criterion = p['criterion'], min_samples_split = p['min_samples_split'],
                                     min_samples_leaf = p['min_samples_leaf'], n_jobs = p['n_jobs'])
        if name == "NN":
            cv_result = deep_crossval(x, y, kfold)
        else:
            cv_result = cross_val_score(model, x, y, cv = kfold, scoring="accuracy")
        results.append(cv_result)
        names.append(name)
        message = "%s: %f (%f)"%(name, cv_result.mean(), cv_result.std())
        print(message)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("cross validation score boxplot")
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def deep_crossval(x, y, kfold):
    cv_result = np.array([])
    for train, test in kfold.split(x, y):
        m = NN(shape = shape, n_classes=n_classes)
        train_x, train_y = x[train], y[train]
        test_x, test_y = x[test], y[test]

        m.fit(train_x, train_y, epochs = 15, verbose = 0)
        result = np.argmax(m.predict(test_x), axis =1)


        cv_result = np.append(cv_result, accuracy_score(test_y, result))

    return cv_result



def grid_search(x, y, num_folds):
    for m, p in zip(models, params):
        if p == {}:
            best_param.append({})
            continue
        grid = GridSearchCV(m[1], p, cv= num_folds, return_train_score=True)
        grid.fit(x, y)

        print(m[0]+" best parameter: ", grid.best_params_)
        best_param.append(grid.best_params_)

def fitting(x, y):
    for name, model in models:
        if name == "NN":
            model = model.fit(x, y, epochs = 15, verbose = 0)
        else:
            model = model.fit(x, y)

def testing(x_test, y_test):
    plt.figure()
    for i, (name, model) in enumerate(models):
        local_predict = model.predict(x_test)
        if name == "NN":
            local_predict = np.argmax(local_predict, axis = 1)
        acc = accuracy_score(y_test, local_predict)
        print(name+": "+ str(acc))
        predict.append(local_predict)

        plt.bar(name, acc)
        plt.text(acc+3, i+.25, str(acc), color = 'red', fontweight = 'bold')
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.show()

def roc(x_test, y_test):
    for name, model in models:
        if name == "NN":
            prob = model.predict(x_test)
            fpr, tpr, th = roc_curve(y_test, prob, pos_label=0)
        else:
            prob = model.decision_function(x_test)
            fpr, tpr, th = roc_curve(y_test, prob[:, 0], pos_label=0)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw = 2, label = name+"(area = %0.4f)"%auc_score)
    plt.plot([0, 1], [0, 1], lw = 2, linestyle = '--')
    plt.xlim([0.0, 1.0])

    plt.xlabel("False positive rate")
    plt.ylabel("True positive Rate")

    plt.title("ROC curve")
    plt.legend()
    plt.show()

def confusionmatrix(y_test):
    for (name, _), p in zip(models, predict):
        cm = confusion_matrix(y_test, p)

        fig, ax = plt.subplots(figsize = (5, 5))
        ax.matshow(cm, cmap = plt.cm.Reds, alpha = 0.3)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x = j, y = i, s = cm[i, j], va = 'center', ha = 'center')
        plt.xlabel("predicted values")
        plt.ylabel("Actual Values")
        plt.title(name)
        plt.show()

def featrue_importance(feature_name):
    plt.figure()
    plt.bar(feature_name, models[7][1].feature_importances_)
    plt.xlabel("Label")
    plt.ylabel("Importance score")
    plt.title("Feature importances")
    plt.show()


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    grid_search(X_train, y_train, 10)
    cross_val_boxplot(X_train, y_train, 10)
    fitting(X_train, y_train)
    featrue_importance(iris.feature_names)
    testing(X_test, y_test)
    confusionmatrix(y_test)
    roc(X_test, y_test)
