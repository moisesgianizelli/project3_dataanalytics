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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder




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

    #  missing values
    cleanRow = readFile.dropna(subset=['Location'])

    # string format 
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

    plt.figure(figsize=(10, 6))
    plt.bar(topFive.index, topFive.values, color='skyblue', alpha=0.7, label='Top 5 Locations')

    for i, value in enumerate(topFive.values):
        plt.text(i, value + 1, f"{percentage[i]}%", ha='center', va='bottom', fontsize=8)

    plt.xlabel('Location')
    plt.ylabel('Number of Records')
    plt.title('Top 5 Locations with Fewest Records')
    plt.legend()
    plt.show()


#task1()    
    
def task2():

    """
    Objective: 
    - Analyse relationship between air pressure and subsequent rainfall.

    Context:
    - High air pressure generally indicates stable weather conditions with a lower chance of rain.
    - High-pressure systems are associated with descending air, which suppresses cloud formation and precipitation.
    - Low-pressure systems are associated with more unstable atmospheric conditions, potentially leading to cloud formation and precipitation, making rain more likely.

    Dataset:
    -  Air pressure recorded at two distinct times: 9 am and 3 pm.

    Task:
    - Validate the assertion that air pressure affects subsequent rainfall.
    - Investigate whether a decrease in pressure might lead to an increased chance of rainfall the following day.

    Strategy:
    - Extract rows with the minimum difference (D) in air pressure between 9 am and 3 pm.
    - Repeat this process 12 times for different values of D in the range [1, 12].

    Analysis:
    - For each D, calculate the number of rainy days and non-rainy days.
    - Divide the number of rainy days by the number of non-rainy days.
    - Repeat this process for each D (1 to 12).
    - Generate Figure 1, likely a plot showing the relationship between pressure difference (D) and the likelihood of rainfall.

    Hypotheses:
    - Hypothesis 1: If there is a significant decrease in air pressure (higher D), the likelihood of rainfall might increase the following day.
    - Hypothesis 2: There might be a non-linear relationship between pressure difference and the chance of rainfall, potentially exhibiting a threshold effect.
    """
    
    hypothesis1 = []
    hypothesis2 = []

    # Iterate through different values of D
    for D in range(1, 13):
        # Extract rows with the minimum difference D in pressure and calculate the number of rainy and non-rainy days
        rows = readFile.loc[abs(readFile['Pressure9am'] - readFile['Pressure3pm']) == D]
        rainyDays = rows[rows['RainTomorrow'] == 'Yes'].shape[0]
        nonRainyDays = rows[rows['RainTomorrow'] == 'No'].shape[0]

        # Hypothesis 1
        ratio1 = rainyDays / nonRainyDays if nonRainyDays != 0 else 0
        hypothesis1.append({'D': D, 'RainyToNonRainyRatio': ratio1})

        # Hypothesis 2
        ratio2 = rainyDays / nonRainyDays if nonRainyDays != 0 and D > 5 else 0
        hypothesis2.append({'D': D, 'RainyToNonRainyRatio': ratio2})

    # Convert the results to DataFrames
    result1 = pd.DataFrame(hypothesis1)
    result2 = pd.DataFrame(hypothesis2)

    # Results for Hypothesis 1
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(result1['D'], result1['RainyToNonRainyRatio'], marker='o', linestyle='-')
    plt.xlabel('Pressure Difference (D)')
    plt.ylabel('Rainy to Non-Rainy Ratio')
    plt.title('Effect of Pressure Difference on Rainfall (Hypothesis 1)')
    plt.grid(True)

    # Results for Hypothesis 2
    plt.subplot(1, 2, 2)
    plt.plot(result2['D'], result2['RainyToNonRainyRatio'], marker='o', linestyle='-')
    plt.xlabel('Pressure Difference (D)')
    plt.ylabel('Rainy to Non-Rainy Ratio')
    plt.title('Effect of Pressure Difference on Rainfall (Hypothesis 2)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Display the results
    print("Results for Hypothesis 1:")
    print(result1)
    print("\nResults for Hypothesis 2:")
    print(result2)

    """
    Hypothese 1: 
    - The initial part of the graph shows a decreasing trend in the ratio of rainy to non-rainy days as the pressure difference (D) increases. 
    - This suggests that a smaller pressure difference is associated with a higher likelihood of rain.
    - Then There is a sudden and significant increase in the ratio around D=8. This sharp increase indicates a notable change in the relationship between pressure difference and rainfall likelihood.
    A higher ratio suggests a substantial increase in the likelihood of rain for this range of pressure differences.

    Hypothese 2:
    Similar to Hypothesis 1, Hypothesis 2 also suggests the presence of a threshold effect. The sharp increase around D=8 followed by a drop indicates that the relationship between pressure difference and rainfall likelihood undergoes a significant change at this point.
    The findings in Hypothesis 2 provide additional evidence for the presence of a threshold effect in the relationship between air pressure differences and rainfall likelihood. The stable ratio within the range D=1 to D=5 suggests a consistent relationship, while the sharp increase and subsequent drop around D=8 indicate a significant change in this relationship.

    
    we conclue that it has an inverse relationship between the pressure difference (D) and the likelihood of rainfall.
    For smaller pressure differences (D=1 to D=4), there is a higher likelihood of rainfall.
    As the pressure difference increases, the likelihood of rainfall decreases.
    """

        
#task2()

def task3():

    attributes = ['WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Temp9am', 'Temp3pm', 'RainTomorrow']
    otherDf = readFile[attributes]

    # Exclude non-numeric columns when calculating the mean
    numcollumns = otherDf.select_dtypes(include=['number']).columns
    means = otherDf[numcollumns].mean(skipna=True)

    # Replace non-numeric columns with NaN in the mean calculation
    for col in otherDf.columns:
        if col not in numcollumns:
            means[col] = float('nan')

    # Fill NaN values with the mean of each column
    otherDf = otherDf.fillna(means)

    # Separate features (X) and target variable (y)
    X = otherDf.drop('RainTomorrow', axis=1)
    y = otherDf['RainTomorrow']
    label_encoder = LabelEncoder()
    yencoded = label_encoder.fit_transform(y)

    # Initialize a decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    columns = ['Max Depth'] + X.columns.tolist()
    results = pd.DataFrame(columns=columns)

    # loop
    upper_range = 36
    for i in range(1, upper_range):
        clf.set_params(max_depth=i)   
        clf.fit(X, yencoded)
        row = [i] + list(clf.feature_importances_)
        results.loc[len(results) + 1] = row

    plt.figure(figsize=(10, 6))

    for feature in results.columns.drop('Max Depth'):
        plt.plot(results['Max Depth'], results[feature], label=feature)

    plt.xlabel('Max Depth')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importances vs Max Depth of Decision Tree')
    plt.legend()
    plt.tight_layout()
    plt.show()

    """
    Executing the code a graphic appears, illustrating the correlation between feature importance and the maximum depth of the three.
    - It indicates that the "Humidity3pm" feature significantly outweighs others in predicting the likelihood of rainfall on the following day
    - About maximum depth, it shows singularly holds importance, signified by a value of 1.
    - The decision tree is singularly focused on a solitary decision. As the maximum depth increases, the influence of humidity at 3 pm gradually diminishes.
    """


#task3()        

def task4():

    # Create a sub-dataset with the specified attributes
    otherDataSet = readFile[[
        'WindDir9am',
        'WindDir3pm',
        'Pressure9am',
        'Pressure3pm',
        'RainTomorrow'
    ]]

    # first question (a)
    xpressure = otherDataSet[['Pressure9am', 'Pressure3pm']]
    y = otherDataSet['RainTomorrow']
    label_encoder = LabelEncoder()
    yencoded = label_encoder.fit_transform(y)

    # Split the dataset
    xpressureTrainning, xtestPressure, ytrain, ytest = train_test_split(xpressure, yencoded, test_size=0.33, random_state=42)

    # Decision Tree Classifier and train the model
    pressureModel = DecisionTreeClassifier(random_state=42, max_depth=20)
    pressureModel.fit(xpressureTrainning, ytrain)

    # Print
    print("\nUsing Pressure as Ordinary Attributes:")
    print("Training Accuracy:", f"{pressureModel.score(xpressureTrainning, ytrain):.5f}")
    print("Test Accuracy:", f"{pressureModel.score(xtestPressure, ytest):.5f}")

    # second question (b)
    # repeat the process
    xwindDirection = otherDataSet[['WindDir9am', 'WindDir3pm']]
    encoder = OneHotEncoder(sparse=False)
    xwindEncoded = encoder.fit_transform(xwindDirection)

    # Split the dataset again
    xtrainWind, xtestWind, ytrain, ytest = train_test_split(xwindEncoded, yencoded, test_size=0.33, random_state=42)

    # Decision Tree and train model
    windModelDir = DecisionTreeClassifier(random_state=42, max_depth=20)
    windModelDir.fit(xtrainWind, ytrain)

    # Print 
    print("\nUsing Wind Direction as Ordinary Attributes:")
    print("Training Accuracy:", f"{windModelDir.score(xtrainWind, ytrain):.5f}")
    print("Test Accuracy:", f"{windModelDir.score(xtestWind, ytest):.5f}")


    """
    Using Pressure as Ordinary Attributes:
    - Training Accuracy: 0.77585
    - Test Accuracy: 0.75170

     Using Wind Direction as Ordinary Attributes:
    - Training Accuracy: 0.75984
    - Test Accuracy: 0.75547

    Considering the test accuracy as a key metric, the model utilizing Wind Direction attributes seems more effective for predicting RainTomorrow
    compared to the model using Pressure attributes.
    """
    

#task4()
def task5():
    # Create a sub-DataFrame with more diverse samples
    otherDf = pd.DataFrame({
        'RainTomorrow': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'WindDir9am': ['NW', 'W', 'E', 'NW', 'NE', 'SW', 'SE', 'S', 'N', 'W'],
        'WindGustDir': ['W', 'E', 'W', 'NW', 'NE', 'SW', 'SE', 'S', 'N', 'W'],
        'WindDir3pm': ['NW', 'E', 'SW', 'NW', 'NE', 'S', 'N', 'W', 'SW', 'SE']
    })

    otherDf['RainTomorrow'] = otherDf['RainTomorrow'].astype('category')
    otherDf = pd.get_dummies(otherDf, columns=['WindDir9am', 'WindGustDir', 'WindDir3pm'], drop_first=True)

    # Lists to store training and test accuracy for Decision Tree Classifier
    trainAcc = []
    trainTestAcc = []

    # Loop over different depths for Decision Tree Classifier
    for depth in range(1, 11):
        # Initialize Decision Tree Classifier and perform cross-validation and store averages
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        trainScore = cross_val_score(tree, otherDf.drop('RainTomorrow', axis=1), otherDf['RainTomorrow'], cv=5)
        trainAcc.append(trainScore.mean())
        scores = cross_val_score(tree, otherDf.drop('RainTomorrow', axis=1), otherDf['RainTomorrow'], cv=5)
        trainTestAcc.append(scores.mean())

    # Lists to store training and test accuracy for KNeighborsClassifier
    trainAcc = []
    testAcc = []

    # Loop over different numbers of neighbors for KNeighborsClassifier
    for neighbors in range(1, 11):
        # Initialize KNeighborsClassifier
        knn_clf = KNeighborsClassifier(n_neighbors=neighbors)

        # Perform cross-validation and store averages
        trainScore = cross_val_score(knn_clf, otherDf.drop('RainTomorrow', axis=1), otherDf['RainTomorrow'], cv=5)
        trainAcc.append(trainScore.mean())

        scores = cross_val_score(knn_clf, otherDf.drop('RainTomorrow', axis=1), otherDf['RainTomorrow'], cv=5)
        testAcc.append(scores.mean())

    # Generate plots
    plt.figure(figsize=(10, 8))

    # Decision Tree 
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 11), trainAcc, label='Decision Tree Training Accuracy')
    plt.plot(range(1, 11), trainTestAcc, label='Decision Tree Test Accuracy')
    plt.title('Decision Tree Classifier Accuracy vs. Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()

    # KNeighborsClassifier 
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 11), trainAcc, label='KNN Training Accuracy')
    plt.plot(range(1, 11), testAcc, label='KNN Test Accuracy')
    plt.title('KNeighbors Classifier Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    """
    This question follows same format as the previous one, the goal here is to perform a machine learning experiment using the Decision Tree Classifier and the K-Nearest Neighbors (KNN) Classifier on a sub-DataFrame containing specific attributes.
    -	Loop over different depths (1 to 10) for Decision Tree Classifier.
    -	Initialize the Decision Tree Classifier with the specified depth.
    -	Perform cross-validation
    Repeat the process for KNeighborsClassifier Training and print them to compare the trend of training and test accuracies.
    The output tho I dont know if its wrong because decision tree graph shows two lines and Kneighbors graph shows only one (a junction of those two lines). My interpretation is somehow a improved training accuracy and KNN lines could highlight a depth value where the Decision Tree's overfitting starts. But I might be wrong because I didnt get all requirements in this task.

    """

#task5()

def task6():

    # Select the specified columns
    columns = ['MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm',
                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                         'Rainfall', 'Temp9am', 'Temp3pm']

    # new data frame and handle any non-numerical values
    newDataFrame = readFile[columns]
    newDataFrame = newDataFrame.apply(pd.to_numeric, errors='coerce')
    newDataFrame = newDataFrame.dropna()

    # standard scaling to the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(newDataFrame)

    # question a - Apply K-Means clustering on the dataset using various numbers of clusters
    cluster = range(2, 9)
    inertias = []

    for clusters in cluster:
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        kmeans.fit(scaled)
        inertias.append(kmeans.inertia_)

    # (b) Utilize an appropriate visualization method to determine the optimal number of clusters
    plt.plot(cluster, inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()

    """
    The plot shows the trade-off between the number of clusters and the within-cluster sum of squares (Inertia).
    # As K increases, Inertia generally decreases, as each cluster becomes more tightly packed.
    # The "elbow" in the plot represents a point where the rate of decrease of Inertia slows down, indicating diminishing returns.

    K=2 to K=3: Inertia decreases from 1 to 0.85
    K=3 to K=4: Inertia decreases from 0.85 to 0.76
    K=4 to K=5: Inertia decreases from 0.76 to 0.74
    K=5 to K=6: Inertia decreases from 0.76 to 0.68
    K=6 to K=7: Inertia decreases from 0.68 to 0.64
    K=7 to K=8: Inertia decreases from 0.64 to 0.60

    the elbow point is between K2 and K3 (decrease significantly)

    """

#task6()
    
    
    
def task7():

    """
    I decided go for temperature prediction for many reason: weather forecasting, energy management, agriculture and tourism for example.
    This approach we can capture relationships between input features and the target variable, allowing for accurate predictions.
    Feature selection considers meteorological factors known to influence temperature.
    The interpretability of regression models provides insights into the relationship between selected features and temperature.
    """
   # Relevant features
    features = ['MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'RainToday']
    target = 'MaxTemp'
    file = readFile[features + [target]].dropna()
    X = file[features]
    y = file[target]

    # Split the dataset 
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    columns = ["RainToday"]
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), features[:-1]),
            ('cat', OneHotEncoder(), columns)
        ])
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ])
    pipeline.fit(xtrain, ytrain)

    # Predictions
    ypred = pipeline.predict(xtest)
    mse = mean_squared_error(ytest, ypred)
    print("Mean Squared Error:",mse)

    plt.scatter(ytest, ypred)
    plt.xlabel("Actual MaxTemp")
    plt.ylabel("Predicted MaxTemp")
    plt.title("Actual vs. Predicted MaxTemp")
    plt.show()

    """
    The output: Small dots form a crescent line, it implies a positive correlation between the predicted and actual temperatures, we conclue that this correlation suggests that the machine learning model, trained on the provided dataset, is performing well in predicting temperatures.
    """

task7()





        