# Importing the necessary libraries for this project
# Pandas and NumPy are for data manipulation
# Streamlit is for creating a web app
# Sklearn libraries are for machine learning tasks like preprocessing, model building, and evaluation
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset
# Make sure to replace the file path with your actual CSV file location
# This dataset seems to be about machine failures
data = pd.read_csv('ai4i2020.csv')

# Preprocessing

# Separating the data into features (X) and target (y)
# The target here is 'Machine_failure', which we'll try to predict
X = data.drop('Machine_failure', axis=1)  # Features
y = data['Machine_failure']  # Target

# We have both categorical and numerical data, so we'll handle them separately
# Identifying the columns
# 'Type' is categorical, while the others are numerical
categorical_cols = ['Type']
numerical_cols = ['Air_temperature__K_', 'Process_temperature__K_', 'Rotational_speed__rpm_', 'Torque__Nm_', 'Tool_wear__min_']

# Creating a preprocessing pipeline for handling both numerical and categorical columns
# Standardizing numerical values, One-hot encoding categorical ones
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),  # Standardize numerical columns
        ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode the categorical column
    ])

# Building the full pipeline for the logistic regression model
# First step will be preprocessing, followed by fitting the classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),  # Preprocessing step
    ('classifier', LogisticRegression())  # Logistic Regression for prediction
])

# Splitting the data into training and testing sets
# 80% for training and 20% for testing, keeping the randomness consistent with random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model with the training data
model.fit(X_train, y_train)

# Testing the model with the testing data
y_pred = model.predict(X_test)

# Calculating accuracy of the model by comparing predictions to actual results
accuracy = accuracy_score(y_test, y_pred)

# Building the Streamlit App

# Displaying the title of our app
st.title("Machine Failure Prediction")

# Section for users to input feature values
st.header("Input Features")

# Creating interactive input fields for all the features using Streamlit's widgets
# The users will select or enter values, which we'll use to make predictions
Type = st.selectbox("Type", ["M", "L"])  # Categorical input for machine type
Air_temperature__K_ = st.number_input("Air Temperature (K)", min_value=290.0, max_value=350.0, value=300.0)  # Numerical input for air temperature
Process_temperature__K_ = st.number_input("Process Temperature (K)", min_value=300.0, max_value=400.0, value=310.0)  # Numerical input for process temperature
Rotational_speed__rpm_ = st.number_input("Rotational Speed (rpm)", min_value=1000.0, max_value=2000.0, value=1500.0)  # Numerical input for rotational speed
Torque__Nm_ = st.number_input("Torque (Nm)", min_value=20.0, max_value=100.0, value=50.0)  # Numerical input for torque
Tool_wear__min_ = st.number_input("Tool Wear (min)", min_value=0.0, max_value=300.0, value=50.0)  # Numerical input for tool wear time

# Arranging the input data into a DataFrame to match our model's expected format
input_data = pd.DataFrame({
    'Type': [Type],  # User's choice for machine type
    'Air_temperature__K_': [Air_temperature__K_],  # User input for air temperature
    'Process_temperature__K_': [Process_temperature__K_],  # User input for process temperature
    'Rotational_speed__rpm_': [Rotational_speed__rpm_],  # User input for rotational speed
    'Torque__Nm_': [Torque__Nm_],  # User input for torque
    'Tool_wear__min_': [Tool_wear__min_]  # User input for tool wear
})

# Using the trained model to make a prediction based on user inputs
prediction = model.predict(input_data)

# Displaying the prediction result
st.subheader("Prediction")
if prediction[0] == 1:
    st.markdown("Machine has failed.")  # If prediction is 1, machine is predicted to have failed
else:
    st.markdown("Machine is working fine.")  # If prediction is 0, machine is predicted to be fine

# Displaying the accuracy of the model for user reference
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")  # Showing the accuracy of the model on test data
