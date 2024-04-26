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
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle

# Check if the model file exists
model_file = "model.h5"
if os.path.exists(model_file):
    # Load the model
    model = load_model(model_file)
    st.write("Model loaded successfully")
else:
    st.error("Model file not found. Make sure model.h5 exists.")

# Check if the tokenizer file exists
if os.path.exists("tokenizer.pickle"):
    # Load the tokenizer
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    st.write("Tokenizer loaded successfully")
else:
    st.error("Tokenizer file not found. Make sure tokenizer.pickle exists.")

# Streamlit app
def main():
    st.title("Hate Speech Detection")

    # Input text area for user input
    text_input = st.text_area("Enter text:", "")

    # Button to trigger hate speech detection
    if st.button("Detect Hate Speech"):
        # Perform hate speech detection
        prediction = predict_hate_speech(text_input)
        if prediction == 0:
            st.write("Predicted Class: Hate Speech")
        elif prediction == 1:
            st.write("Predicted Class: Offensive Language")
        else:
            st.write("Predicted Class: Neither")

# Function to preprocess text and perform hate speech detection
def predict_hate_speech(text):
    # Preprocess the input text
    text = preprocess_text(text)  # Add your preprocessing function here if needed
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')

    # Debug statements to print the generated sequence
    st.write("Generated sequence:", sequence)
    st.write("Padded sequence:", padded_sequence)

    # Perform hate speech detection
    prediction = model.predict(padded_sequence)

    # Convert prediction to class label
    return prediction.argmax()

if __name__ == "__main__":
    main()



