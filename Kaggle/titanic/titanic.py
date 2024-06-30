import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('Kaggle\\titanic\\model.json')

# Define a function to preprocess input data
def preprocess_data(input_data):
    """
    Encodes categorical features in the input data.

    Args:
        input_data (pd.DataFrame): DataFrame containing input data with 
                                     categorical features 'Gender' and 'Embarked'.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    input_data['Gender'] = input_data['Gender'].map({'Male': 0, 'Female': 1})
    input_data['Embarked'] = input_data['Embarked'].map({'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2})
    return input_data

# --- Streamlit App ---
st.title('Titanic Survival Prediction App')

st.write('''
## Overview
This app predicts the likelihood of survival for passengers on the Titanic using a trained XGBoost model. 
Provide passenger details in the sidebar to see the survival prediction.
''')

# --- Sidebar for user input ---
st.sidebar.title('Input Fields')
st.sidebar.write('Enter passenger details:')

pclass = st.sidebar.selectbox('Ticket class', [1, 2, 3])
gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
age = st.sidebar.slider('Age in years', min_value=1, max_value=100, value=18)
sibsp = st.sidebar.slider('Siblings/Spouses Aboard', min_value=0, max_value=8, value=0)
parch = st.sidebar.slider('Parents/Children Aboard', min_value=0, max_value=6, value=0)
fare = st.sidebar.slider('Fare', min_value=0.0, max_value=512.3292, value=14.4542, format="%.2f")
embarked = st.sidebar.selectbox('Embarked Location', ['Cherbourg', 'Queenstown', 'Southampton'])

# Create a dictionary to hold the input data
data = {
    'Pclass': pclass,
    'Gender': gender,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}

# Create a DataFrame from the input data
input_data = pd.DataFrame(data, index=[0])

# Preprocess the input data
input_data = preprocess_data(input_data)

# --- Display the preprocessed input data ---
st.write('## Input Data')
st.write(input_data)

# --- Make predictions ---
prediction = model.predict(input_data)[0]

# --- Display the prediction ---
st.write('## Prediction:', prediction)
if prediction == 1:
    st.write('Passenger is likely to survive.')
else:
    st.write('Passenger is likely to not survive.')

# --- Model Accuracies Comparison ---
model_accuracies = {
    "Logistic Regression": 0.7988826815642458,
    "Random Forest": 0.8156424581005587,
    "Support Vector Machine": 0.8212290502793296,
    "Gradient Boosting Classifier": 0.8156424581005587,
    "K-Nearest Neighbors": 0.8044692737430168,
    "Decision Tree": 0.7821229050279329,
    "AdaBoost": 0.8044692737430168,
    "Gaussian Naive Bayes": 0.770949720670391,
    "Stochastic Gradient Descent": 0.7821229050279329,
    "XGBoost_1": 0.8379888268156425,
    "XGBoost_2": 0.8435754189944135
}
df = pd.DataFrame(list(model_accuracies.items()), columns=['Model', 'Accuracy'])

st.title('Model Accuracies Comparison')
plt.figure(figsize=(10, 6))
plt.barh(df['Model'], df['Accuracy'])
plt.xlabel('Accuracy')
plt.ylabel('Model')
plt.title('Model Accuracies Comparison')
st.pyplot(plt)

# --- Feature Heatmap ---
heatmap_data = {
    'Survived': [1.000000, -0.338481, 0.543351, -0.069809, -0.035322, 0.081629, 0.257307, 0.106811],
    'Pclass': [-0.338481, 1.000000, -0.131900, -0.331339, 0.083081, 0.018443, -0.549500, 0.045702],
    'Sex': [0.543351, -0.131900, 1.000000, -0.084153, 0.114631, 0.245489, 0.182333, 0.116569],
    'Age': [-0.069809, -0.331339, -0.084153, 1.000000, -0.232625, -0.179191, 0.091566, 0.007461],
    'SibSp': [-0.035322, 0.083081, 0.114631, -0.232625, 1.000000, 0.414838, 0.159651, -0.059961],
    'Parch': [0.081629, 0.018443, 0.245489, -0.179191, 0.414838, 1.000000, 0.216225, -0.078665],
    'Fare': [0.257307, -0.549500, 0.182333, 0.091566, 0.159651, 0.216225, 1.000000, 0.062142],
    'Embarked': [0.106811, 0.045702, 0.116569, 0.007461, -0.059961, -0.078665, 0.062142, 1.000000]
}
heatmap_df = pd.DataFrame(heatmap_data)

st.title('Feature Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_df.corr(), annot=True, cmap='coolwarm', fmt=".4f", ax=ax)
st.pyplot(fig) 

# --- About Me & Links Section ---
st.markdown("---")
st.header("About the Creator")
st.write("This app was created by Spagestic. You can connect with me on:")

# Replace with your actual links
st.write(f"[GitHub](https://github.com/spagestic)  |  [LinkedIn](https://linkedin.com/in/vishalginni)") 

st.write("Check out the source code for this app on [GitHub](https://github.com/Spagestic/Data-Science-Projects/tree/main/Kaggle/titanic).") 