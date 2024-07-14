import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time


ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load pre-trained model and TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set up Streamlit app
st.title("Spam Email/SMS Detector")

# User input for message
input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        # Display spinner while predicting
        with st.spinner('Predicting...'):
            time.sleep(2)  # Simulate a delay for prediction
            # Preprocess and predict
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Display result with animation
            st.subheader("Prediction Result")
            if result == 1:
                st.error("🚫 Spam")
            else:
                st.success("✅ Not Spam")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit by - Vineet")