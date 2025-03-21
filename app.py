pip freeze > requirements.txt
streamlit==1.4.0
joblib==1.0.1
numpy==1.20.3
scikit-learn==0.24.2

import streamlit as st
import joblib
import numpy as np

# Load the model and vectorizer
model = joblib.load("spam_classifier_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("Spam Message Detector 📩")
st.write("Enter a message to check if it's spam or ham!")

# User Input
message = st.text_area("Type your message here:")

if st.button("Classify Message"):
    if message:
        # Transform input text
        message_tfidf = tfidf.transform([message])
        
        # Predict probabilities
        spam_prob = model.predict_proba(message_tfidf)[0][1]  # Probability of spam
        ham_prob = 1 - spam_prob  # Probability of ham
        
        # Display results
        st.write(f"### Spam Probability: {spam_prob * 100:.2f}%")
        st.write(f"### Ham Probability: {ham_prob * 100:.2f}%")
        
        if spam_prob > 0.5:
            st.error("⚠️ This message is likely SPAM!")
        else:
            st.success("✅ This message is likely HAM (not spam).")
    else:
        st.warning("Please enter a message to classify.")
