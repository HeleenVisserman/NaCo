from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

rstate = 1


bCancer = load_breast_cancer()
x3, y3 = bCancer['data'], bCancer['target']

x_train, x_test, y_train, y_test = train_test_split(x3, y3, random_state=rstate)

def tree():
    clf1 = DecisionTreeClassifier(random_state=rstate)
    clf1.fit(x_train, y_train)
    return clf1.score(x_test, y_test)

def adaBoost(estimators, tree_depth):
    clf2 = AdaBoostClassifier(n_estimators= estimators, base_estimator=DecisionTreeClassifier(max_depth=tree_depth))
    clf2.fit(x_train, y_train)
    return clf2.score(x_test,y_test)

def plot_accuracy_n_estimators():
    estimators = [x for x in range(1,101)]
    scores = []
    for e in estimators:
        scores.append(adaBoost(e, None))

    plt.plot(estimators, scores, label = "AdaBoost")
    plt.plot(estimators, [tree()]*len(scores), label = "Decision Tree")
    plt.xlabel("Number of Estimators for AdaBoost")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.show()

def plot_accuracy_depth_estimators():
    depths = [x for x in range(1,101)]
    scores = []
    for d in depths:
        scores.append(adaBoost(50, d))

    plt.plot(depths, scores, label = "AdaBoost")
    plt.plot(depths, [tree()]*len(scores), label = "Decision Tree")
    plt.xlabel("Allowed tree depth of estimators")
    plt.ylabel("Accuracy Score")
    plt.legend()
    plt.show()

plot_accuracy_depth_estimators()
plot_accuracy_n_estimators()


