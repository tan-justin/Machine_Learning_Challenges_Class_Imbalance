'''
Name: Justin Tan
Assignment: Give your models a grade
Date: Feb 10 2024
File: generate_profile.py
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO

'''
Type: Class
Name: GenerateProfile
Purpose: Class to contain methods for generating a data profile
Parameters: training data (Pandas Dataframe)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: prep_data
Purpose: Generate a dictionary that contains the statistics of each feature in the X set
Parameters: None
Output: A dictionary of stats that can be of type Int, float, etc.
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_histogram
Purpose: Generate a histogram of each feature in the x set
Parameters: Number of bins, file output path
Output: Histograms of each feature (.png)
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_pdf
Purpose: Generate a pdf where each page in the pdf corresponds to a feature in the X set and the Y set. Each page contains information
         on the feature such as statistics and a histogram
Parameters: file output path
Output: pdf file
---------------------------------------------------------------------------------------------------------------------------------
Type: Function
Name: generate_pie_chart
Purpose: Generate a pie chart for the class label so that we can see the class distribution
Parameters: file output path
Output: Pie chart of class label distribution (.png)
'''

class GenerateProfile:

    def __init__(self, data):

        self.data = data

    def prep_data(self):

        data = self.data.copy()
        feature_labels_name = data.columns.tolist()
        class_labels_name = feature_labels_name.pop(-1)
        num_features = len(feature_labels_name)
        stat_dict = {}
        for columns in feature_labels_name:

            mean = data[columns].mean()
            median = data[columns].median()
            min = data[columns].min()
            max = data[columns].max()

            stat_dict[columns] = {

                'type': data[columns].dtype,
                'mean': mean,
                'median': median,
                'min': min,
                'max': max,
                            
            }

        return stat_dict
    
    def generate_histogram(self, bins = 10, output_path = 'histogram'):

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        data = self.data.copy()
        for columns in data.columns:

            data_without_null = data[columns].dropna()
            plt.hist(data_without_null, bins = bins, edgecolor = 'black')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.title(f'Histogram for {columns}')
            output_file = os.path.join(output_path, f'{columns}_histogram.png')
            plt.savefig(output_file)
            plt.close()
            print(f'Histogram saved at: {output_file}')

    def generate_pdf(self, output_path='data_profile.pdf'):

        buffer = BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=letter)
        try:

            for column, stats in self.prep_data().items():

                pdf.drawString(15, 750, f"Feature: {column}")
                pdf.drawString(15, 730, f"Type: {stats['type']}")
                pdf.drawString(15, 710, f"Mean: {stats['mean']}")
                pdf.drawString(15, 690, f"Median: {stats['median']}")
                pdf.drawString(15, 670, f"Max: {stats['max']}")
                pdf.drawString(15, 650, f"Min: {stats['min']}")
                histogram_path = os.path.join('histogram', f'{column}_histogram.png')
                pdf.drawInlineImage(histogram_path, 10, 200, width=480, height=360)
                pdf.showPage()

        finally:
            pie_path = os.path.join('pie chart',f'{self.data.copy().columns.tolist().pop(-1)}_pie_chart.png')
            pdf.drawInlineImage(pie_path, 10, 200, width = 480, height = 360)
            pdf.showPage()
            pdf.save()

        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())

        print(f'Combined PDF report saved at: {output_path}')

    def generate_pie_chart(self, output_path = 'pie chart'):

        data = self.data.copy()
        feature_labels_name = data.columns.tolist()
        class_labels_name = feature_labels_name.pop(-1)
        if not os.path.exists(output_path):

            os.makedirs(output_path)

        class_data = data[class_labels_name]
        class_instance_count = class_data.value_counts()
        plt.pie(class_instance_count, labels = class_instance_count.index, startangle=90)
        labels = [f'{category}: {count / class_instance_count.sum() * 100:.1f}%' for category, count in class_instance_count.items()]
        plt.legend(labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.title(f'Categorical distribution of {self.data.copy().columns.tolist().pop(-1)}')
        output_file = os.path.join(output_path, f'{class_labels_name}_pie_chart.png')
        plt.savefig(output_file)
        plt.close()
        print(f'Pie chart saved at: {output_file}')

