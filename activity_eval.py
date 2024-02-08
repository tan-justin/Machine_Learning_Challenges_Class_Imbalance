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
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

class ActivityEval:

    def __init__(self, data, test):
        y_label = data.copy().iloc[:,-1]
        x_label = data.copy().iloc[:,:-1]
        self.person_label = None
        if 'person' in data:
            person = data.copy()['person']
            self.person_label = person
            x_label.drop(columns = ['person'])
        self.y_label = y_label
        self.x_label = x_label
        y_test = test.copy().iloc[:,-1]
        x_test = test.copy().iloc[:,:-1]
        self.person_test = None
        if 'person' in test:
            person = test.copy()['person']
            self.person_test = person
            x_test.drop(columns = ['person'])
        self.y_test = y_test
        self.x_test = x_test
        self.data = data
        self.method_accuracy = {}
        self.actual_accuracy = {}
        self.test = test

    def methodA(self): #80-20 train test split 
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label is not None:
            person_label = self.person_label.copy()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

        decision_tree = DecisionTreeClassifier(random_state = 0)
        decision_tree.fit(x_train, y_train)
        y_pred_dt = decision_tree.predict(x_test)
        decision_tree_accuracy = accuracy_score(y_test, y_pred_dt)

        random_forest = RandomForestClassifier(random_state = 0)
        random_forest.fit(x_train, y_train)
        y_pred_rf = random_forest.predict(x_test)
        random_forest_accuracy = accuracy_score(y_test, y_pred_rf)

        k_neighbors = KNeighborsClassifier(n_neighbors = 3)
        k_neighbors.fit(x_train, y_train)
        y_pred_kn = k_neighbors.predict(x_test)
        k_neighbors_accuracy = accuracy_score(y_test, y_pred_kn)

        mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
        mlp.fit(x_train, y_train)
        y_pred_mlp = mlp.predict(x_test)
        mlp_accuracy = accuracy_score(y_test, y_pred_mlp)

        accuracy = {}
        accuracy['decision tree'] = decision_tree_accuracy
        accuracy['random forest'] = random_forest_accuracy
        accuracy['3 nn'] = k_neighbors_accuracy
        accuracy['MLP'] = mlp_accuracy

        self.method_accuracy['A'] = accuracy

    def methodB(self): #10 fold cross validation kfold(shuffle = true)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label is not None:
            person_label = self.person_label.copy()
        kf = KFold(n_splits = 10, shuffle = True, random_state = 0)
        classifiers = ['decision tree','random forest','3 nn','MLP']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for classifier in classifiers:
                if classifier == 'decision tree':
                    decision_tree = DecisionTreeClassifier(random_state = 0)
                    decision_tree.fit(x_train, y_train)
                    y_pred_dt = decision_tree.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dt)
                if classifier == 'random forest':
                    random_forest = RandomForestClassifier(random_state = 0)
                    random_forest.fit(x_train, y_train)
                    y_pred_rf = random_forest.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_rf)
                if classifier == '3 nn':
                    k_neighbors = KNeighborsClassifier(n_neighbors = 3)
                    k_neighbors.fit(x_train, y_train)
                    y_pred_kn = k_neighbors.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_kn)
                if classifier == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
                    mlp.fit(x_train, y_train)
                    y_pred_mlp = mlp.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_mlp)
        for classifier in avg_accuracy:
            avg_accuracy[classifier] /= 10
        self.method_accuracy['B'] = avg_accuracy

    def methodC(self): #class stratified cross validation stratifiedkfold(shuffle = True)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label is not None:
            person_label = self.person_label.copy()
        skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 0)
        classifiers = ['decision tree','random forest','3 nn','MLP']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            for classifier in classifiers:
                if classifier == 'decision tree':
                    decision_tree = DecisionTreeClassifier(random_state = 0)
                    decision_tree.fit(x_train, y_train)
                    y_pred_dt = decision_tree.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dt)
                if classifier == 'random forest':
                    random_forest = RandomForestClassifier(random_state = 0)
                    random_forest.fit(x_train, y_train)
                    y_pred_rf = random_forest.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_rf)
                if classifier == '3 nn':
                    k_neighbors = KNeighborsClassifier(n_neighbors = 3)
                    k_neighbors.fit(x_train, y_train)
                    y_pred_kn = k_neighbors.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_kn)
                if classifier == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
                    mlp.fit(x_train, y_train)
                    y_pred_mlp = mlp.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_mlp)
        for classifier in avg_accuracy:
            avg_accuracy[classifier] /= 10
        self.method_accuracy['C'] = avg_accuracy

    def methodD(self): #groupwise 10 fold cross validation groupkfold()
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label is not None:
            person_label = self.person_label.copy()
        gkf = GroupKFold(n_splits = 10)
        classifiers = ['decision tree','random forest','3 nn','MLP']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train_index, test_index in gkf.split(x, y, person_label):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for classifier in classifiers:
                if classifier == 'decision tree':
                    decision_tree = DecisionTreeClassifier(random_state = 0)
                    decision_tree.fit(x_train, y_train)
                    y_pred_dt = decision_tree.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dt)
                if classifier == 'random forest':
                    random_forest = RandomForestClassifier(random_state = 0)
                    random_forest.fit(x_train, y_train)
                    y_pred_rf = random_forest.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_rf)
                if classifier == '3 nn':
                    k_neighbors = KNeighborsClassifier(n_neighbors = 3)
                    k_neighbors.fit(x_train, y_train)
                    y_pred_kn = k_neighbors.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_kn)
                if classifier == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
                    mlp.fit(x_train, y_train)
                    y_pred_mlp = mlp.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_mlp)
        for classifier in avg_accuracy:
            avg_accuracy[classifier] /= 10
        self.method_accuracy['D'] = avg_accuracy
        
    def methodE(self): #class stratified, groupwise 10 fold cross validation stratifiedgroupkfold(shuffle = true)
        y = self.y_label.copy()
        x = self.x_label.copy()
        if self.person_label is not None:
            person_label = self.person_label.copy()
        sgkf = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state = 0)
        classifiers = ['decision tree','random forest','3 nn','MLP']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for train_index, test_index in sgkf.split(x, y, person_label):
            x_train, x_test = x.iloc[train_index], x.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            for classifier in classifiers:
                if classifier == 'decision tree':
                    decision_tree = DecisionTreeClassifier(random_state = 0)
                    decision_tree.fit(x_train, y_train)
                    y_pred_dt = decision_tree.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dt)
                if classifier == 'random forest':
                    random_forest = RandomForestClassifier(random_state = 0)
                    random_forest.fit(x_train, y_train)
                    y_pred_rf = random_forest.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_rf)
                if classifier == '3 nn':
                    k_neighbors = KNeighborsClassifier(n_neighbors = 3)
                    k_neighbors.fit(x_train, y_train)
                    y_pred_kn = k_neighbors.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_kn)
                if classifier == 'MLP':
                    mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
                    mlp.fit(x_train, y_train)
                    y_pred_mlp = mlp.predict(x_test)
                    avg_accuracy[classifier] += accuracy_score(y_test, y_pred_mlp)
        for classifier in avg_accuracy:
            avg_accuracy[classifier] /= 10
        self.method_accuracy['E'] = avg_accuracy

    def final_method(self):
        y_train = self.y_label.copy()
        x_train = self.x_label.copy()
        y_test = self.y_test.copy()
        x_test = self.x_test.copy()
        if self.person_label is not None:
            person_train = self.person_label.copy()
        if self.person_test is not None:
            person_test = self.person_test.copy()
        classifiers = ['decision tree','random forest','3 nn','MLP', 'dummy']
        avg_accuracy = {}
        for classifier in classifiers:
            avg_accuracy[classifier] = 0
        for classifier in classifiers:
            if classifier == 'decision tree':
                decision_tree = DecisionTreeClassifier(random_state = 0)
                decision_tree.fit(x_train, y_train)
                y_pred_dt = decision_tree.predict(x_test)
                avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dt)
            if classifier == 'random forest':
                random_forest = RandomForestClassifier(random_state = 0)
                random_forest.fit(x_train, y_train)
                y_pred_rf = random_forest.predict(x_test)
                avg_accuracy[classifier] += accuracy_score(y_test, y_pred_rf)
            if classifier == '3 nn':
                k_neighbors = KNeighborsClassifier(n_neighbors = 3)
                k_neighbors.fit(x_train, y_train)
                y_pred_kn = k_neighbors.predict(x_test)
                avg_accuracy[classifier] += accuracy_score(y_test, y_pred_kn)
            if classifier == 'MLP':
                mlp = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, random_state = 0)
                mlp.fit(x_train, y_train)
                y_pred_mlp = mlp.predict(x_test)
                avg_accuracy[classifier] += accuracy_score(y_test, y_pred_mlp)
            if classifier == 'dummy':
                dummy = DummyClassifier(strategy = 'stratified')
                dummy.fit(x_train, y_train)
                y_pred_dummy = dummy.predict(x_test)
                avg_accuracy[classifier] += accuracy_score(y_test, y_pred_dummy)

        self.actual_accuracy = avg_accuracy

    def methodology(self):
        self.methodA()
        self.methodB()
        self.methodC()
        self.methodD()
        self.methodE()
        print('train-test methods:')
        method_accuracy = pd.DataFrame(self.method_accuracy)
        print(method_accuracy)
        self.final_method()
        print('actual accuracy:')
        actual_accuracy = pd.DataFrame([self.actual_accuracy]).T
        actual_accuracy = actual_accuracy.rename(columns = {0: 'Actual_Accuracy'})
        print(actual_accuracy)
        classifiers = ['decision tree','random forest','3 nn','MLP']
        real_method = ['A','B','C','D','E']
        signed_error = {}
        for method, values in self.method_accuracy.items():
            placeholder = {}
            for classifier, accuracy in values.items(): # for each classifier method and its associated accuracy
                placeholder[classifier] = accuracy - self.actual_accuracy[classifier] #get the signed difference
            signed_error[method] = placeholder
        avg_signed_error = {}
        for method in real_method:
            avg_signed_error[method] = 0
        for method, values in signed_error.items():
            for classifier, error in values.items():
               avg_signed_error[method] += error
        for method in avg_signed_error:
            avg_signed_error[method] /= len(classifiers)
        table = pd.DataFrame(signed_error).T
        table['Avg'] = avg_signed_error
        print('signed error:')
        print(table.T)

            



