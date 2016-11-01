#coding=utf8
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
tmp=0
dictionary = {'criterion':'', 'splitter':'','max_features':0,'max_depth':0,'min_samples_split':0,'min_samples_leaf':0,'persort':True}
criterion = ("gini", "entropy")
splitter = ("best", "random")
df = pd.read_csv('csvtest.csv')
X = df.values[0:1941, 0:26].tolist()
y = df.values[0:1941, 27].tolist()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
x_train_1,x_test_1,y_train_1,y_test_1=cross_validation.train_test_split(x_train,y_train,test_size=0.2)
for c in criterion:
  for s in splitter:
    for max_features in range(20,27):
        print "max_features: ",max_features
        for max_depth in range(5,15):
            print "max_depth: ", max_depth
            for samples_split in range(2,5):
                print "samples_split: ",samples_split
                for samples_leaf in range(1,5):
                    print "samples_leaf: ", samples_leaf
                    for per in (True,False):
                        model = DecisionTreeClassifier(criterion=c,splitter=s,max_features=max_features,max_depth=max_depth,
                                               min_samples_split=samples_split,min_samples_leaf=samples_leaf,presort=per);
                        model.fit(x_train_1,y_train_1)
                        scores = cross_validation.cross_val_score(model, x_test_1, y_test_1, cv=5)
                        if np.mean(scores)>tmp:
                            dictionary['criterion']=c
                            dictionary['splitter']=s
                            dictionary['max_features']=max_features
                            dictionary['max_depth']=max_depth
                            dictionary['min_samples_split']=samples_split
                            dictionary['min_samples_leaf']=samples_leaf
                            dictionary['persort']=per
print "round completion"
print dictionary
model = DecisionTreeClassifier(criterion= dictionary['criterion'], splitter=dictionary['splitter'], max_features=dictionary['max_features'], max_depth=   dictionary['max_depth'],
                               min_samples_split= dictionary['min_samples_split'], min_samples_leaf=dictionary['min_samples_leaf'], presort=dictionary['persort']);
model.fit(x_train, y_train)
scores = cross_validation.cross_val_score(model, x_test, y_test, cv=10)


print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
feature = ['X_Minimum','X_Maximum','Y_Minimum','Y_Maximum','Pixels_Areas','X_Perimeter','Y_Perimeter','Sum_of_Luminosity','Minimum_of_Luminosity','Maximum_of_Luminosity','Length_of_Conveyer','TypeOfSteel_A300','TypeOfSteel_A400','Steel_Plate_Thickness','Edges_Index,Empty_Index','Square_Index','Outside_X_Index','Edges_X_Index','Edges_Y_Index','Outside_Global_Index','LogOfAreas','Log_X_Index','Log_Y_Index','Orientation_Index','Luminosity_Index','SigmoidOfAreas']
predicted=model.predict(x_test)
expected = y_test
print "精确度：",np.mean(expected == predicted)
with open("CARTtree.dot", 'w') as f:
  f = tree.export_graphviz(model, out_file=f, feature_names=feature )
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

