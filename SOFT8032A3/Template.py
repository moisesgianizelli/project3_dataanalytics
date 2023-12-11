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

        # Hypothesis 2: Check for a threshold effect
        ratio2 = rainyDays / nonRainyDays if nonRainyDays != 0 and D > 5 else 0
        hypothesis2.append({'D': D, 'RainyToNonRainyRatio': ratio2})

    # Convert the results to DataFrames
    result1 = pd.DataFrame(hypothesis1)
    result2 = pd.DataFrame(hypothesis2)

    # Plot the results for Hypothesis 1
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(result1['D'], result1['RainyToNonRainyRatio'], marker='o', linestyle='-')
    plt.xlabel('Pressure Difference (D)')
    plt.ylabel('Rainy to Non-Rainy Ratio')
    plt.title('Effect of Pressure Difference on Rainfall (Hypothesis 1)')
    plt.grid(True)

    # Plot the results for Hypothesis 2
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

    """
   
    """


    depths = list(range(1, 36))
    feature_importance_results = {attribute: [] for attribute in ["WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Temp9am", "Temp3pm"]}

    # Loop over different maximum depths
    for depth in depths:
        # Initialize decision tree classifier with the current maximum depth
        clf = DecisionTreeClassifier(max_depth=depth)
        
        # Use all data as training (no need to split)
        X = readFile.drop(columns=["RainTomorrow"])
        y = readFile["RainTomorrow"]
        
        # Train the model
        clf.fit(X, y)
        
        # Store feature importance for the current depth
        for attribute in feature_importance_results:
            feature_importance_results[attribute].append(clf.feature_importances_[X.columns.get_loc(attribute)])

    # Visualize the results
    plt.figure(figsize=(12, 8))
    for attribute in feature_importance_results:
        plt.plot(depths, feature_importance_results[attribute], label=attribute)

    plt.xlabel('Maximum Depth')
    plt.ylabel('Feature Importance')
    plt.title('Impact of Maximum Depth on Feature Importance in Decision Tree')
    plt.legend()
    plt.show()
 ###########################giving error   
#task3()        
def task4():
    # attributes_pressure = ['Pressure9am', 'Pressure3pm', 'RainTomorrow']
    # attributes_wind = ['WindDir3pm', 'WindDir9am', 'RainTomorrow']

    # # Model using Pressure attributes
    # X_pressure = readFile[attributes_pressure].dropna().drop('RainTomorrow', axis=1)
    # y_pressure = readFile[attributes_pressure].dropna()['RainTomorrow']
    # X_train_pressure, X_test_pressure, y_train_pressure, y_test_pressure = train_test_split(X_pressure, y_pressure, test_size=0.33, random_state=42)

    # # Model using Wind attributes
    # X_wind = readFile[attributes_wind].dropna().drop('RainTomorrow', axis=1)
    # y_wind = readFile[attributes_wind].dropna()['RainTomorrow']
    # X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X_wind, y_wind, test_size=0.33, random_state=42)

    # # Decision Tree Classifier
    # clf_pressure = DecisionTreeClassifier(random_state=42)
    # clf_wind = DecisionTreeClassifier(random_state=42)

    # # Train the models
    # clf_pressure.fit(X_train_pressure, y_train_pressure)
    # clf_wind.fit(X_train_wind, y_train_wind)

    # # Predictions
    # y_pred_train_pressure = clf_pressure.predict(X_train_pressure)
    # y_pred_test_pressure = clf_pressure.predict(X_test_pressure)

    # y_pred_train_wind = clf_wind.predict(X_train_wind)
    # y_pred_test_wind = clf_wind.predict(X_test_wind)

    # # Calculate accuracies
    # accuracy_train_pressure = accuracy_score(y_train_pressure, y_pred_train_pressure)
    # accuracy_test_pressure = accuracy_score(y_test_pressure, y_pred_test_pressure)

    # accuracy_train_wind = accuracy_score(y_train_wind, y_pred_train_wind)
    # accuracy_test_wind = accuracy_score(y_test_wind, y_pred_test_wind)

    # print("Accuracy using Pressure attributes:")
    # print(f"Training Accuracy: {accuracy_train_pressure:.2f}")
    # print(f"Test Accuracy: {accuracy_test_pressure:.2f}\n")

    # print("Accuracy using Wind attributes:")
    # print(f"Training Accuracy: {accuracy_train_wind:.2f}")
    # print(f"Test Accuracy: {accuracy_test_wind:.2f}")
    print("test")
    

#task4()
def task5():
    # Create a sub-DataFrame with more diverse samples
    sub_df = pd.DataFrame({
        'RainTomorrow': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No'],
        'WindDir9am': ['NW', 'W', 'E', 'NW', 'NE', 'SW', 'SE', 'S', 'N', 'W'],
        'WindGustDir': ['W', 'E', 'W', 'NW', 'NE', 'SW', 'SE', 'S', 'N', 'W'],
        'WindDir3pm': ['NW', 'E', 'SW', 'NW', 'NE', 'S', 'N', 'W', 'SW', 'SE']
    })

    # Convert 'RainTomorrow' to a categorical variable
    sub_df['RainTomorrow'] = sub_df['RainTomorrow'].astype('category')

    # One-hot encode categorical variables
    sub_df = pd.get_dummies(sub_df, columns=['WindDir9am', 'WindGustDir', 'WindDir3pm'], drop_first=True)

    # Lists to store training and test accuracy for Decision Tree Classifier
    tree_train_accuracy = []
    tree_test_accuracy = []

    # Loop over different depths for Decision Tree Classifier
    for depth in range(1, 11):
        # Initialize Decision Tree Classifier
        tree_clf = DecisionTreeClassifier(max_depth=depth, random_state=42)

        # Perform cross-validation and store averages
        train_scores = cross_val_score(tree_clf, sub_df.drop('RainTomorrow', axis=1), sub_df['RainTomorrow'], cv=5)
        tree_train_accuracy.append(train_scores.mean())

        test_scores = cross_val_score(tree_clf, sub_df.drop('RainTomorrow', axis=1), sub_df['RainTomorrow'], cv=5)
        tree_test_accuracy.append(test_scores.mean())

    # Lists to store training and test accuracy for KNeighborsClassifier
    knn_train_accuracy = []
    knn_test_accuracy = []

    # Loop over different numbers of neighbors for KNeighborsClassifier
    for neighbors in range(1, 11):
        # Initialize KNeighborsClassifier
        knn_clf = KNeighborsClassifier(n_neighbors=neighbors)

        # Perform cross-validation and store averages
        train_scores = cross_val_score(knn_clf, sub_df.drop('RainTomorrow', axis=1), sub_df['RainTomorrow'], cv=5)
        knn_train_accuracy.append(train_scores.mean())

        test_scores = cross_val_score(knn_clf, sub_df.drop('RainTomorrow', axis=1), sub_df['RainTomorrow'], cv=5)
        knn_test_accuracy.append(test_scores.mean())

    # Generate plots
    plt.figure(figsize=(10, 8))

    # Plot Decision Tree Classifier accuracy
    plt.subplot(2, 1, 1)
    plt.plot(range(1, 11), tree_train_accuracy, label='Decision Tree Training Accuracy')
    plt.plot(range(1, 11), tree_test_accuracy, label='Decision Tree Test Accuracy')
    plt.title('Decision Tree Classifier Accuracy vs. Depth')
    plt.xlabel('Depth')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot KNeighborsClassifier accuracy
    plt.subplot(2, 1, 2)
    plt.plot(range(1, 11), knn_train_accuracy, label='KNN Training Accuracy')
    plt.plot(range(1, 11), knn_test_accuracy, label='KNN Test Accuracy')
    plt.title('KNeighbors Classifier Accuracy vs. Number of Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

########################### fix and take conclusions
task5()

def task6():

    # Select the specified columns
    selected_columns = ['MinTemp', 'MaxTemp', 'WindSpeed9am', 'WindSpeed3pm',
                         'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
                         'Rainfall', 'Temp9am', 'Temp3pm']

    # Create a new DataFrame with selected columns
    selected_df = readFile[selected_columns]

    # Check for and handle any non-numerical values
    selected_df = selected_df.apply(pd.to_numeric, errors='coerce')
    selected_df = selected_df.dropna()

    # Apply standard scaling to the data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(selected_df)

    # (a) Apply K-Means clustering on the dataset using various numbers of clusters
    cluster_range = range(2, 9)
    inertias = []

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scaled_df)
        inertias.append(kmeans.inertia_)

    # (b) Utilize an appropriate visualization method to determine the optimal number of clusters
    plt.plot(cluster_range, inertias, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.show()
####################################### FIXING THE CODE AND TAKE CONCLUSIONS
#task6()
    
    
    
def task7():

    """
    I decided go for temperature prediction for many reason: weather forecasting, energy management, agriculture and tourism for example.
    This approach we can capture relationships between input features and the target variable, allowing for accurate predictions.
    Feature selection considers meteorological factors known to influence temperature.
    The interpretability of regression models provides insights into the relationship between selected features and temperature.
    """
   # Select relevant features and target variable
    features = ['MinTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'RainToday']
    target = 'MaxTemp'
    file = readFile[features + [target]].dropna()
    X = file[features]
    y = file[target]

    # Split the dataset into training and testing sets and create a pipeline
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

    # Predictions on the test set 
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

#task7()





        