import streamlit as st
import joblib
import time
from transformers import pipeline

st.title("Twitter Sentiment Analysis")
def load_model():
    model=pipeline('sentiment-analysis')
    return model
nlp=load_model()
# Loading model

model = joblib.load(open('twitter_sentiment.joblib', 'rb'))

tweet = st.text_input("Enter Your Tweet")

submit = st.button("Predict")

if submit:
    start = time.time()
    prediction = nlp.predict([tweet])
    end = time.time()
    st.write('Prediction time taken:', round(end - start, 2), 'seconds')
    st.write(prediction[0]['label'])
