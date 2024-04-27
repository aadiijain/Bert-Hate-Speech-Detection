# import streamlit as st
# import torch
# from transformers import BertForSequenceClassification, BertTokenizer

# # Load pre-trained model and tokenizer
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# # Function to preprocess input text and make predictions
# def predict_text(input_text):
#     # Tokenize input text
#     input_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    
#     # Make prediction
#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits
#         predicted_class = torch.argmax(logits, dim=1).item()

#     # Map predicted class to label
#     label_map = {0: "Hate speech detected", 1: "Offensive language", 2: "Neither"}
#     predicted_label = label_map[predicted_class]
    
#     return predicted_label

# # Streamlit UI
# def main():
#     st.title("Hate Speech Detection")

#     # Text input
#     input_text = st.text_area("Enter text:", "")

#     # Button to trigger prediction
#     if st.button("Predict"):
#         if input_text.strip() == "":
#             st.error("Please enter some text.")
#         else:
#             # Make prediction
#             predicted_label = predict_text(input_text)
#             st.success(f"Predicted class: {predicted_label}")

# if __name__ == "__main__":
#     main()
# --------------------------------------------------------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import re
import time
import datetime

# Load the pretrained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to preprocess the input text
def preprocess_text(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return text

# Function to encode the input text
def bert_encode(data, max_len):
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

# Function to format elapsed time
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to perform inference on the input text
def evaluate_text(model, text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt', max_length=64, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    predicted_class = np.argmax(logits)
    return predicted_class

# Streamlit app
st.title("Hate Speech Detection with BERT")

# User input for text
text_input = st.text_input("Enter a sentence to check for hate speech:")

if text_input:
    # Preprocess the text
    preprocessed_text = preprocess_text(text_input)

    # Perform inference
    predicted_class = evaluate_text(model, preprocessed_text)

    # Display prediction
    label_mapping = {0: "Non-hate speech", 1: "Hate speech"}
    prediction_label = label_mapping[predicted_class]
    st.write("Prediction:", prediction_label)





