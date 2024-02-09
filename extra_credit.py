'''
Name: Justin Tan
Assignment: Give your models a grade
Date: Feb 10 2024
File: activity_eval.py
'''

import pandas as pd
import numpy as np
from collections import Counter

'''
Type: Class
Name: ExtraCredit
Purpose: Contain the methods required to solve the extra credit question
Parameters: Class Imbalanced Training dataset
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: balance_dataset
Purpose: Creates a class balanced dataset by applying oversampling to the original training dataset
Parameters: None
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: check_class_balance
Purpose: Verifies that the new dataset created is class-balanced
Parameters: dataframe of the new dataset
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: undersample
Purpose: Creates a class balanced dataset by applying undersampling to the original training dataset. The intention is to determine
         if the negative signed difference for the MLP classifier is due to the nature of oversampling or a coincidence
Parameters: None
'''

class ExtraCredit:

    def __init__(self, train):
        self.train = train

    def balance_dataset(self):
        train = self.train.copy()
        feature_labels_name = train.columns
        class_count = Counter(train.iloc[:, -1])
        max_class_count = max(class_count.values())
        balanced = pd.DataFrame(columns = feature_labels_name)
        for class_label in class_count.keys():
            class_samples = train[train.iloc[:, -1] == class_label]
            num_samples_to_add = max_class_count - class_count[class_label]
            if num_samples_to_add > 0:
                random_samples = class_samples.sample(n = num_samples_to_add, replace = True)
                balanced = pd.concat([balanced, random_samples])
            balanced = pd.concat([balanced, class_samples])
        object_columns = balanced.select_dtypes(include = ['object']).columns
        balanced[object_columns] = balanced[object_columns].astype('int64')
        shuffled = balanced.sample(frac = 1).reset_index(drop = True)
        return shuffled
    
    def check_class_balance(self, df):
        class_count = df.iloc[:, -1].value_counts()
        is_balanced = len(set(class_count)) == 1
        return is_balanced, class_count
    
    def undersample(self):
        train = self.train.copy()
        feature_labels_name = train.columns
        class_count = Counter(train.iloc[:,-1])
        min_class_count = min(class_count.values())
        balanced = pd.DataFrame(columns = feature_labels_name)
        for class_label in class_count.keys():
            class_samples = train[train.iloc[:,-1] == class_label]
            undersampled_samples = class_samples.sample(n = min_class_count, replace = False)
            balanced = pd.concat([balanced, undersampled_samples])
        object_columns = balanced.select_dtypes(include = ['object']).columns
        balanced[object_columns] = balanced[object_columns].astype('int64')
        shuffled = balanced.sample(frac = 1).reset_index(drop = True)
        return shuffled
