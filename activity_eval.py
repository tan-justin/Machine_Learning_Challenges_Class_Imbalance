import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split #method A
from sklearn.model_selection import KFold #method B
from sklearn.model_selection import StratifiedKFold #method C
from sklearn.model_selection import GroupKFold #method D
from sklearn.model_selection import StratifiedGroupKFold #method E
from sklearn.metrics import accuracy_score

class ActivityEval:

    def __init__(self, data):
        y_label = data.copy().iloc[:,-1]
        x_label = data.copy().iloc[:,:-1]
        if data['person']:
            print('Person class detected in this assignment, creating a separate profile and dropping it from y_label')
            person = data.copy()['person']
            self.person_label = person
            y_label.drop(columns = ['person'])
        self.y_label = y_label
        self.x_label = x_label
        self.data = data
        self.method_accuracy = {}

    def methodA(self): #80-20 train test split 
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label:
            person_label = self.person_label.copy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)
        y_pred_dt = decision_tree.predict(x_test)
        decision_tree_accuracy = accuracy_score(y_test, y_pred_dt)

        random_forest = RandomForestClassifier(random_state = 0)
        random_forest.fit(x_train, y_train)
        y_pred_rf = random_forest.predict(x_test)
        random_forest_accuracy = accuracy_score(y_test, y_pred_rf)

        k_neighbors = KNeighborsClassifier()
        k_neighbors.fit(x_train, y_train)
        y_pred_kn = k_neighbors.predict(x_test)
        k_neighbors_accuracy = accuracy_score(y_test, y_pred_kn)

        mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000)
        mlp.fit(x_train, y_train)
        y_pred_mlp = mlp.predict(x_test)
        mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

        accuracy = {}
        accuracy['decision tree'] = decision_tree_accuracy
        accuracy['random forest'] = random_forest_accuracy
        accuracy['k neighbors'] = k_neighbors_accuracy
        accuracy['MLP'] = mlp_accuracy

    def methodB(self): #10 fold cross validation kfold(shuffle = true)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label:
            person_label = self.person_label.copy()

        kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
        for train_index, test_index in kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
        pass

    def methodC(self): #class stratified cross validation stratifiedkfold(shuffle = True)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label:
            person_label = self.person_label.copy()
        pass

    def methodD(self): #groupwise 10 fold cross validation groupkfold()
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label:
            person_label = self.person_label.copy()
        pass

    def methodE(self): #class stratified, groupwise 10 fold cross validation stratifiedgroupkfold(shuffle = true)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label:
            person_label = self.person_label.copy()
        pass
    
    def methodology(self):
        self.methodA()
        self.methodB()
        self.methodC()
        self.methodD()
        self.methodE()
        print(self.method_accuracy)
        

        
            


