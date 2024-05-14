# Titanic Survival Prediction

## Introduction
The sinking of the Titanic, one of the deadliest maritime disasters in history, has intrigued researchers and data scientists alike. On April 15, 1912, the RMS Titanic, deemed "unsinkable," tragically collided with an iceberg during its maiden voyage, leading to the loss of 1502 lives out of 2224 passengers and crew. This disaster raises a poignant question: what factors influenced survival rates among the passengers?

## Objective
Our goal is to develop a predictive model that identifies the characteristics of passengers who were more likely to survive the Titanic disaster. By analyzing passenger data such as name, age, gender, and socio-economic status, we aim to uncover patterns that could predict survival outcomes.

## Dataset Overview
This competition provides access to two datasets: `train.csv` and `test.csv`.

### Train Dataset (`train.csv`)
- **Size**: 891 records
- **Columns**: Includes passenger details (name, age, gender, socio-economic class, etc.) and a binary indicator for survival (`Survived`).
- **Purpose**: Serves as the training set to build and validate our predictive models.

### Test Dataset (`test.csv`)
- **Size**: 418 records
- **Columns**: Similar to the train dataset but lacks the `Survived` column.
- **Purpose**: Used to evaluate the performance of our predictive models.

## Task
Your task is to analyze the `train.csv` dataset to identify key predictors of survival and then apply this understanding to predict the survival outcomes for the passengers in the `test.csv` dataset.

## Submission Guidelines
Submit your predictions in a CSV file named `Predictions.csv`. The file should contain exactly 418 entries (excluding the header row) and include two columns:

- `PassengerId`: The ID of the passenger (can be sorted in any order).
- `Survived`: Your predicted survival outcome (1 for survived, 0 for deceased).

## Evaluation
Your predictions will be evaluated based on their accuracy in predicting the survival outcomes of the passengers in the `test.csv` dataset. The closer your predictions align with the actual outcomes, the higher your score on the Kaggle leaderboard.

## Getting Started
Explore the datasets in the "Data" tab to familiarize yourself with the passenger information. Use this data to build a predictive model and submit your predictions to Kaggle to see how your model performs against others.

