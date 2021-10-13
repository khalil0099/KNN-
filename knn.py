# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 08:52:20 2021

@author: HP
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris.csv',header=0)

def euclidean_distance(a,b):
    sum=0
    for i in range (a.size-1):
        sum+=(a[i]-b[i])**2
    return math.sqrt(sum)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:-1],df.iloc[:,-1], test_size=0.2)

prediction = np.zeros(y_test.shape[0],dtype='object')# 150 lignes remplis en  0
print(df.shape[0])# la forme , combien de lignes et de colonnes 
# recherche du plus proche voisin
for i in range(X_test.shape[0]):
    distMin = np.inf # borne inferieur 
    indexMin = -1;# variable fixé à l'avance 
    current = X_test.iloc[i,:]
    for j in range(X_train.shape[0]):
        t = X_train.iloc[j,:]
        dist = euclidean_distance(current,t)
        if dist < distMin:
            distMin = dist
            indexMin = j
    prediction[i] = y_train.iloc[indexMin]# pour remplir les 150 lignes 

# évaluation du classifieur
cnf_matrix = confusion_matrix(prediction, y_test)
print(cnf_matrix)
print(classification_report(prediction, y_test))

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

n_classes = 3
plot_colors = "ryb"
plot_step = 0.02
iris = load_iris()
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)

    plt.subplot(2, 3, pairidx + 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
plt.suptitle("Decision surface of a 3-NN using paired features")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
plt.show()


import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.metrics import confusion_matrix,classification_report,precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris.csv',header=0)
X_train,X_test,y_tain,y_test=train_test_split(df.iloc[:,0:-1],df.iloc[:,-1],test_size=0.2)
#boucle sur le nombre de voisin
perf=[]

for k in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    predictions=knn.predict(X_test) 
    perf.append(precision_score(predictions,y_test,average='micro'))
plt.plot(perf)
plt.show()
