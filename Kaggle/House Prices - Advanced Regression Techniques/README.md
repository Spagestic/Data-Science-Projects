# House Prices: Advanced Regression Techniques

## Overview
Start here if you have some experience with R or Python and machine learning basics. This competition is ideal for data science students looking to expand their skill set before advancing to a featured competition.

## Getting Started
Take advantage of the starter notebook to jumpstart your analysis.

## Competition Description
This playground competition requires you to predict the final price of homes in Ames, Iowa, using 79 explanatory variables. It's not just about the number of bedrooms or a white-picket fence—every detail can influence the price negotiations.

## Evaluation
### Goal
Your task is to predict the sales price for each house. Specifically, you'll provide the `SalePrice` for each `Id` in the test set.

### Metric
Submissions are evaluated on the Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. This ensures that errors in predicting both expensive and cheap houses will affect the result equally.
