# Employee Attrition Prediction System

## Project Overview
The Employee Attrition Prediction System is a Machine Learning project that predicts whether an employee is likely to leave an organization. It uses employee-related data and a Random Forest Classifier to identify patterns associated with employee attrition.

## Objectives
- Analyze employee data using Python and Pandas.
- Perform data cleaning and preprocessing.
- Convert categorical data into numerical data using One-Hot Encoding.
- Train a Machine Learning model using Random Forest.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Dataset
The dataset contains employee-related information, including:
- Age and Gender
- Department and Job Role
- Education and Marital Status
- Business Travel
- Overtime
- Monthly Income
- Job Satisfaction
- Years at the Company
- Attrition status

The target variable is `Attrition`, which indicates whether an employee left the organization.

## Machine Learning Model
A Random Forest Classifier is used to train the model and predict employee attrition.

The dataset is divided into training and testing sets to evaluate the model on unseen data.

## Model Evaluation
The model is evaluated using:
- Accuracy Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

The initial model achieved approximately **83.67% accuracy** on the test dataset. However, the model's ability to identify employees who leave should also be evaluated using recall and F1-score for the attrition class.

## Project Structure

```text
Employee-Attrition-Prediction-System/
├── prediction system.py
├── employee pred.csv
├── requirements.txt
└── README.md