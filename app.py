import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download required nltk data
@st.cache_resource
def load_nltk():
    nltk.download("stopwords")

load_nltk()

# preprocessing function
def text_transform(text):
    stop_words = set(stopwords.words('english'))

    # simple tokenization (avoids punkt dependency)
    trans_text = text.lower().split()

    # remove stopwords
    trans_text = [x for x in trans_text if x not in stop_words]

    # remove punctuation / special characters
    trans_text = [x for x in trans_text if x.isalnum()]

    # stemming
    stemming = PorterStemmer()
    trans_text = [stemming.stem(x) for x in trans_text]

    return " ".join(trans_text)


# load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# UI
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter your message")

if st.button("Predict"):
    if input_sms.strip() != "":
        transformed_sms = text_transform(input_sms)

        # vectorize
        vector = tfidf.transform([transformed_sms])

        # predict
        result = model.predict(vector)[0]

        # display result
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")