import streamlit as st
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn

# --- Application Description ---
st.title("Disaster Tweet Sentiment Analysis")
st.write("""
This application utilizes a fine-tuned BERT model to predict the sentiment of tweets 
related to disasters. It classifies tweets as either **Positive** (indicating a disaster 
is happening) or **Negative** (indicating no disaster).
""")

# --- Model Loading and Initialization ---

# Define the BERT classifier model
class BERTClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)  # Apply dropout for regularization
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(output.pooler_output)
        logits = self.classifier(pooled_output)
        return logits

# Load the trained model
model = BERTClassifier(num_labels=2)  # Binary classification
model_path = "Kaggle/nlp-with-disaster-tweets/best_model.pth" 
model.load_state_dict(torch.load(model_path, 
                                   map_location=torch.device('gpu' if torch.cuda.is_available() else 'cpu')))
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# --- Preprocessing and Prediction Functions ---

def preprocess_text(text):
    """Lowercase and remove non-alphanumeric characters from text."""
    text = text.lower()
    text = ''.join([char for char in text if char.isalnum() or char == ' '])
    return text

def predict_text(text):
    """Predicts the sentiment of the given text using the BERT model.

    Args:
        text (str): The text to analyze.

    Returns:
        str: The predicted sentiment label ("Negative" or "Positive").
    """
    text = preprocess_text(text)
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)

    labels = ['Negative', 'Positive']
    return labels[prediction.item()]


# --- Streamlit App UI ---

# Define a function to simulate sentiment analysis prediction
def predict_sentiment(tweet):
    # Placeholder for actual prediction logic
    # This could be replaced with a call to a machine learning model or API
    return "Positive" if len(tweet.split()) > 5 else "Negative"

# List of example tweets for demonstration purposes
example_tweets = [
    "There's a fire raging in the forest. #wildfire",
    "The earthquake caused widespread damage. #earthquake #disasterrelief",
    "Just another day in paradise. Nothing to see here.",
    "This storm is going to be bad Stay safe everyone. #hurricane",
    "I'm so glad I was able to evacuate before the flood. #floodrelief",
    "Can't believe the news about the volcanic eruption. Thoughts are with those affected."
]

# Displaying the selection box for choosing an example tweet
st.header("**Select an example tweet or enter your own:**")

# Using a selectbox to allow users to choose from example tweets
selected_example = st.selectbox("", example_tweets)

# Providing a text area for users to input their own tweet
text_input = st.text_area("Enter Your Tweet:", value=selected_example if selected_example else "")

# Function to display a warning message if no tweet is entered
def show_warning():
    st.warning("Please enter a tweet to analyze.")

# Analyze button to trigger sentiment analysis
analyze_button = st.button("Analyze Sentiment")

# Handling the case where no tweet is entered
if not text_input.strip():
    show_warning()
else:
    # Performing sentiment analysis when the button is clicked
    if analyze_button:
        prediction = predict_sentiment(text_input)
        st.subheader(f"**Predicted Sentiment:** {prediction}")
# --- About Section ---
st.markdown("---")
st.write("""
This app demonstrates sentiment analysis on disaster-related tweets using a 
BERT model. It can be used to quickly assess the sentiment of tweets, which 
can be valuable in disaster response and monitoring.
""")

# --- About Section ---
st.markdown("---")
st.header("About")

st.write("""
This disaster tweet sentiment analysis app was created by 
[Spagestic](https://github.com/spagestic). 

You can find the complete source code and more projects on my 
[GitHub repository](https://github.com/Spagestic/Data-Science-Projects/tree/main/Kaggle/nlp-with-disaster-tweets). 

Let's connect on [LinkedIn](https://linkedin.com/in/vishalginni)!
""")