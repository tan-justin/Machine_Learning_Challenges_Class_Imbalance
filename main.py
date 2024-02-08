import pandas as pd
from generate_profile import GenerateProfile
from activity_eval import ActivityEval

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

if __name__ =="__main__":
    main()
