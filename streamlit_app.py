import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to preprocess input text and make predictions
def predict_text(input_text):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=64, truncation=True, padding='max_length', return_tensors='pt')
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    # Map predicted class to label
    label_map = {0: "Hate speech detected", 1: "Offensive language", 2: "Neither"}
    predicted_label = label_map[predicted_class]
    
    return predicted_label

# Streamlit UI
def main():
    st.title("Hate Speech Detection")

    # Text input
    input_text = st.text_area("Enter text:", "")

    # Button to trigger prediction
    if st.button("Predict"):
        if input_text.strip() == "":
            st.error("Please enter some text.")
        else:
            # Make prediction
            predicted_label = predict_text(input_text)
            st.success(f"Predicted class: {predicted_label}")

if __name__ == "__main__":
    main()

