from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Load data
data = load_breast_cancer()
print('A Machine Learning Model using K-Nearest_Neighbours For Breast Cancer Classification')
print("Feature names:")
print(data.feature_names)
print("Target names:")
print(data.target_names)

# Split data
x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size=0.2)

# Train model
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
print("Model accuracy on test data:", clf.score(x_test, y_test))

# --- Prediction from User Input ---

# Print instruction for user input:
print("\nEnter the following feature values separated by commas.")
print("You must enter exactly", len(data.feature_names), "values in the following order:")
print(", ".join(data.feature_names))

# Get user input (example: "14.0,20.0,90.0,...")
user_input = input("Enter values: ")

# Split and convert to float
try:
    feature_values = [float(val.strip()) for val in user_input.split(",")]
except Exception as e:
    print("Error converting input values to float:", e)
    exit(1)

if len(feature_values) != len(data.feature_names):
    print("Error: Expected", len(data.feature_names), "values but got", len(feature_values))
    exit(1)

# Predict
prediction = clf.predict([feature_values])
print("The model predicts:", data.target_names[prediction[0]])
