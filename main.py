'''
Name: Justin Tan
Assignment: Give your models a grade
Date: Feb 10 2024
File: main.py
'''

import pandas as pd
from generate_profile import GenerateProfile
from activity_eval import ActivityEval
from extra_credit import ExtraCredit

'''
Type: Function
Name: read_csv_data
Purpose: Convert a csv file into a Pandas Dataframe
Parameters: csv file path (string)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: main
Purpose: Driver code to run the program
Parameters: None
'''

def read_csv_data(file_path):
    return pd.read_csv(file_path)
    
def main():
    csv_file_path = 'activity-dev.csv'
    test_file_path = 'activity-heldout.csv'
    train_data = read_csv_data(csv_file_path)
    test_data = read_csv_data(test_file_path)
    data_instance = GenerateProfile(train_data)
    data_instance.generate_histogram()
    data_instance.generate_pie_chart()
    data_instance.generate_pdf()

    train_test_instance = ActivityEval(train_data, test_data)
    train_test_instance.methodology()

    print('Extra Credit starts here')
    print('Oversampling: ')
    extra_credit_gen_instance = ExtraCredit(train_data)
    balanced = extra_credit_gen_instance.balance_dataset()
    balanced.to_csv('oversampled_dataset.csv', index = False)
    is_balanced, class_counts = extra_credit_gen_instance.check_class_balance(balanced)
    if is_balanced:
        print("dataset is class-balanced.")
    else:
        print("dataset is not class-balanced.")
    extra_credit_instance = ActivityEval(balanced, test_data)
    extra_credit_instance.methodology()
    print('Undersampling: ')
    undersampled = extra_credit_gen_instance.undersample()
    under_balanced, class_under = extra_credit_gen_instance.check_class_balance(undersampled)
    if under_balanced:
        print("dataset is class-balanced")
    else:
        print("dataset is not class-balanced")
    undersampled.to_csv('undersampled_dataset.csv', index = False)
    under_instance = ActivityEval(undersampled, test_data)
    under_instance.methodology()

if __name__ =="__main__":
    main()
