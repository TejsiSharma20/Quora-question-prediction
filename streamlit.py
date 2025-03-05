import streamlit as st
import joblib
import numpy as np
import re
import string

# Load the trained model
model = joblib.load("model.pkl")

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\\d+", "", text)
    return text

# Function to extract numerical features
def extract_features(q1, q2):
    q1_clean = clean_text(q1)
    q2_clean = clean_text(q2)
    return np.array([
        len(q1_clean.split()),  # Word count of first question
        len(q2_clean.split()),  # Word count of second question
        len(set(q1_clean.split()) & set(q2_clean.split()))  # Common words count
    ]).reshape(1, -1)

# Streamlit UI
st.title("Quora Duplicate Question Prediction")
st.write("Enter two questions to check if they are duplicates.")

# User input fields
question1 = st.text_area("Enter first question:")
question2 = st.text_area("Enter second question:")

# Predict button
if st.button("Predict"):
    if question1 and question2:
        # Extract numerical features
        input_data = extract_features(question1, question2)
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display result
        if prediction == 1:
            st.success("These questions are duplicates! ✅")
        else:
            st.warning("These questions are not duplicates. ❌")
    else:
        st.error("Please enter both questions.")


