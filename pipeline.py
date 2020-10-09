from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
from deep import NN
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

shape = (1000, 100)
seed = 100

models = []
models.append(['LR', LogisticRegression()])
models.append(["LDA", LinearDiscriminantAnalysis()])
models.append(["KNN", KNeighborsClassifier()])
models.append(["CART", DecisionTreeClassifier()])
models.append(["NB", GaussianNB()])
models.append(["SVM", SVC()])
models.append(['XB', XGBClassifier()])
models.append(["RFC", RandomForestClassifier()])
models.append(["NN", NN(shape, 1)])

def cross_val_boxplot(x, y, num_folds):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=num_folds)
        cv_result = cross_val_score(model, x, y, cv = kfold, scoring="accuracy")
        results.append(cv_result)
        names.append(name)
        message = "%s: %f (%f)"%(name, cv_result.mean(), cv_result.std())
        print(message)

    fig = plt.figure()
    fig.subtitle("cross validation score boxplot")
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

