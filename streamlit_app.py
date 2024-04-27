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
import time
import datetime
import re

# Load the pretrained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define label mapping
label_mapping = {0: "Not", 1: "Hate speech", 2: "Offensive language"}

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

# Function to perform inference on the input sentence
def predict_sentence(model, sentence):
    model.eval()
    # Preprocess the text
    preprocessed_sentence = preprocess_text(sentence)
    # Tokenize and encode the text
    MAX_LEN = 64
    input_ids, attention_masks = bert_encode([preprocessed_sentence], MAX_LEN)
    # Convert to PyTorch datatypes
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    # Perform inference
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
    logits = outputs[0]
    predicted_class = np.argmax(logits.numpy(), axis=1)[0]
    return label_mapping[predicted_class]

# Streamlit app
st.title("Hate Speech Detection with BERT")

sentence = st.text_input("Enter a sentence to check for hate speech:")
if sentence:
    prediction_label = predict_sentence(model, sentence)
    st.write(f"Predicted Label: {prediction_label}")





