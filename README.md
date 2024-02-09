Machine Learning Challenges


Description: 

This program is designed to showcase the different methods of handling imbalanced data. The evaluation methods utilized in the code are as follows:

A: Train-Test Split
B: 10-Fold Cross-Validation
C: Stratified 10-Fold Cross-Validation
D: Groupwise 10-Fold Cross-Validation
E: Stratified-Groupwise 10-Fold Cross-Validation

The csv files are sourced and then modified from the following website: https://archive.ics.uci.edu/dataset/427/activity+recognition+with+healthy+older+people+using+a+batteryless+wearable+sensor

The program will first generate a data profile of the features found in activity-dev.csv to understand the imbalances of the data. It then performs the 5 above evaluation methods. For each method, 4 different classifiers are used to determine how effective the evaluation methodology is for handling datasets with imbalanced datasets. The 4 classifiers used are:

1. Decision Trees
2. Random Forest
3. K-Nearest Neighbors
4. Multi-Layered Perceptrons

After that, the program trains each of the 4 classifiers on the entire training set. The activity-heldout.csv file contains the test dataset. The program evaluates the effectiveness of each of the 4 classifiers using the test dataset. The class refers to the 'activity' column.  It also trains a dummy classifier to act as a baseline for determining if the classifiers are effective. After this, the program determines the best classifier and prints out the associated confusion matrix. activity_eval.py will print out the following tables:

1. Estimated error grouped by evaluation methodologies used and classifier used
2. Actual accuracy of the classifiers
3. The best classifier and the best classifier's confusion matrix
4. Signed error differences between the estimated error and the actual accuracy

In extra_credit.py, the program will generate 2 new class-balanced datasets using the imbalanced training set. The methods used for generating the 2 new class-balanced datasets are as follows:

1. Oversampling
2. Undersampling

Oversampling involves finding the class with the most number of items, then increasing the population of other classes via random sampling items within the other classes. This continues until all classes have the same number of items.

Undersampling involves finding the class with the least number of items, then decreasing the population of other classes via random sampling within each class. This continues until all classes have the same number of items. 

The program then feeds back the new datasets as training sets for activity_eval.py. 

Instructions:

To run the program, clone the repository and open the directory. Run the program in zsh using python3 main.py


Packages required: 

Numpy, Pandas, Scikit-Learn, ReportLab, MatPlotLib

Links to install these packages:

Numpy: https://numpy.org/install/

Pandas: https://pandas.pydata.org/docs/getting_started/install.html

Scikit-Learn: https://scikit-learn.org/stable/install.html

MatPlotLib: https://matplotlib.org/stable/users/installing/index.html

ReportLab: https://pypi.org/project/reportlab/


Credits: 

This was one of the homework assignment associated with the Machine Learning Challenges (Winter 2024 iteration) course at Oregon State University. All credits belong to the course developer, Dr. Kiri Wagstaff and the lecturer-in-charge, Prof. Rebecca Hutchinson. All code is written by myself solely for the purpose of the assignment. For more information on the course developer and lecturer-in-charge:

Dr. Kiri Wagstaff: https://www.wkiri.com/

Prof. Rebecca Hutchinson: https://hutchinson-lab.github.io/


Use: 

The code shall be used for personal educational purposes only. Students of current (Winter 2024) and future iterations of Machine Learning Challenges at Oregon State University may not use any code in this repo for this assignment should this or a similar assignment be assigned. If any Oregon State University student is found to have plagarized any code in this repo, the author of the repository cannot be held responsible for the incident of plagarism. The author promises to cooperate in any investigations regarding plagarism pertaining to this repo if required. If any of the code in this repo is reused for strictly personal projects, please credit this repository. 
