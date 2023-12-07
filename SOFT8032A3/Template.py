#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 29/11/2023


Name: Moises Munaldi
Student ID: R00225292
Cohort: evSD3

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

readFile = pd.read_csv('SOFT8032A3/weather.csv')

# Use the below few lines to ignore the warning messages

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
def warn(*args, **kwargs):
    pass
warnings.warn = warn




def  task1():
   
    """
    Find the number of unique locations present in the dataset. Utilize an
    appropriate visualization technique to display the five locations with the
    fewest records or rows. Present the percentage for each section and perform all necessary data cleaning.
    """

    # if there is missing values
    cleanRow = readFile.dropna(subset=['Location'])

    # string format if necessary
    # cleanRow['Location'] = cleanRow['Location'].astype(str)

    # Reformating the data
    cleanRow['Location'] = cleanRow['Location'].str.strip().str.upper()
    location = cleanRow['Location'].value_counts()
    topFive = location.nsmallest(5)

    totalRecords = len(cleanRow)
    percentage = (topFive / totalRecords) * 100
    percentage = percentage.round(2)

    print("Top 5:")
    print(topFive)
    print("***********")
    print("Percentage:")
    print(percentage)


#task1()    
    
def task2():

    """
    analyze the effect of the difference in air pressure between 9 am and 3 pm and rainfall the next day
    """
    
   # Drop rows with missing values in 'Pressure9am' and 'Pressure3pm'
    cleanRow = readFile.dropna(subset=['Pressure9am', 'Pressure3pm', 'RainTomorrow'])

    # Calculate the difference in pressure between 9 am and 3 pm
    cleanRow['PressureDiff'] = cleanRow['Pressure9am'] - cleanRow['Pressure3pm']

    # Create lists to store results
    pressure_diffs = []
    rainy_to_non_rainy_ratios = []

    # Iterate over the range of differences [1, 12]
    for D in range(1, 13):
        # Extract rows with the minimum difference D
        min_diff_rows = cleanRow[cleanRow['PressureDiff'] == D]

        # Calculate the number of rainy days and non-rainy days
        rainy_days = min_diff_rows[min_diff_rows['RainTomorrow'] == 'Yes'].shape[0]
        non_rainy_days = min_diff_rows[min_diff_rows['RainTomorrow'] == 'No'].shape[0]

        # Calculate the ratio of rainy to non-rainy days
        ratio = rainy_days / non_rainy_days if non_rainy_days != 0 else 0

        # Append results to lists
        pressure_diffs.append(D)
        rainy_to_non_rainy_ratios.append(ratio)

    # Create a bar plot
    plt.bar(pressure_diffs, rainy_to_non_rainy_ratios, color='blue', alpha=0.7)
    plt.xlabel('Minimum Pressure Difference (D)')
    plt.ylabel('Number of Rainy Days / Number of Non-Rainy Days')
    plt.title('Effect of Pressure Difference on Rainfall')
    plt.xticks(pressure_diffs)
    plt.show()

################################## WRONG #######################################
        
#task2()

def task3():
    
    
        
#def task4():
    
    


#def task5():
    
    



#def task6():
    
    
    
#def task7():
    





        