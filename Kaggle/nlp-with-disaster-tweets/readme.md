# Real or Not? NLP with Disaster Tweets

This beginner-friendly project is perfect for data scientists looking to dive into the world of Natural Language Processing (NLP). The dataset is manageable, and you can do all the work within Kaggle Notebooks!

## Project Description

In times of emergency, Twitter has become a crucial communication channel. However, distinguishing between tweets reporting real disasters and those using disaster-related words metaphorically can be challenging for machines.

**Your Challenge:** Build a machine learning model to predict whether a tweet reports a real disaster or not. You'll work with a dataset of 10,000 hand-classified tweets.

## Dataset

- **train.csv:** The labeled training set (tweet text, location, keyword, and target - whether it's about a real disaster).
- **test.csv:** The unlabeled test set (make predictions on these!).
- **sample_submission.csv:** A sample submission file showing the correct format.

**Note:** The dataset contains text that some may find offensive.

## Your Task

Predict whether a tweet describes a real disaster (1) or not (0).

**Columns:**

- **id:** Unique tweet identifier.
- **text:** The tweet's text.
- **location:** The tweet's location (may be blank).
- **keyword:** A keyword from the tweet (may be blank).
- **target:** (Only in train.csv) Indicates a real disaster (1) or not (0).

## Getting Started

1. Clone this repository.
2. Download the dataset files from the competition page.
3. Explore the data, preprocess the text, and start building your NLP model!
