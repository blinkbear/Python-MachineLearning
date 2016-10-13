import numpy as np
import pandas as pd
from sklearn import cross_validation
df = pd.read_csv('csvtest.csv')
X = df.values[0:1941, 0:26].tolist()
y = df.values[0:1941, 27].tolist()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
# fit a CART model to the data
model = DecisionTreeClassifier(criterion='gini', max_depth=12)
scores = cross_validation.cross_val_score(model, X, y, cv=10)
print np.mean(scores)
# model.fit(x_train, y_train)
# expected = y_train
# predicted = model.predict(x_train)
# print np.mean(expected == predicted)
# # make predictions
# expected = y_test
# predicted = model.predict(x_test)
# # summarize the fit of the model
#
# print np.mean(expected == predicted)
#     if max < accury:
#         j=i
#         max = accury
# print j
# print max
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))