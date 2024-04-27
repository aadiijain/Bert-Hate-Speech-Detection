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

# Function to perform inference on the test set
def evaluate(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    t0 = time.time()
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.append(logits)
        true_labels.append(label_ids)
    elapsed = format_time(time.time() - t0)
    return predictions, true_labels, elapsed

# Streamlit app
st.title("Hate Speech Detection with BERT")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)

    # Preprocess the text
    df['tweet'] = df['tweet'].apply(preprocess_text)

    # Tokenize and encode the text
    MAX_LEN = 64
    input_ids, attention_masks = bert_encode(df['tweet'].values, MAX_LEN)

    # Convert to PyTorch datatypes
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    labels = torch.tensor(df['class'].values)

    # Create DataLoader
    batch_size = 32
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Perform inference
    predictions, true_labels, elapsed_time = evaluate(model, prediction_dataloader)
    st.write(f"Inference time: {elapsed_time}")

    # Calculate accuracy
    flat_predictions = np.concatenate(predictions, axis=0)
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = np.concatenate(true_labels, axis=0)
    accuracy = np.sum(flat_predictions == flat_true_labels) / len(flat_true_labels)
    st.write(f"Accuracy: {accuracy}")




