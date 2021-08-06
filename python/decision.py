import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from six import StringIO
import pydot
import pickle
from datetime import datetime

dataset = pd.read_csv("../csv/decisiontree.csv")
test = pd.read_csv("../csv/test.csv")
X = dataset.drop('Category', axis=1)
y = dataset['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = DecisionTreeClassifier()
toPrint = classifier.fit(X_train, y_train)

# y_pred = classifier.predict(X_test)
y_pred = classifier.predict(test)
print(y_pred)
dateTimeObj = datetime.now()
date = str(dateTimeObj.year) + '-' + str(dateTimeObj.month) + '-' + str(dateTimeObj.day) + '-' + str(dateTimeObj.hour) + '-' + str(dateTimeObj.minute) + '-' + str(dateTimeObj.second)
filename = '../model/finalized_model_tree'+ date +'.sav'
pickle.dump(classifier, open(filename, 'wb'))

dot_data = StringIO()
list = ("Hours","Movement","Day")
classList = ("Restaurant","Niente","Svago","Sport")
tree.export_graphviz(toPrint, out_file=dot_data,feature_names=list,class_names=classList)
graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph[0].write_pdf("../graph/decisiontree.pdf")  # must access graph's first element


# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
#
# print(tree.plot_tree(toPrint))
