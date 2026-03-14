import streamlit as st
import pickle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def text_transform(text):

    stop_words = set(stopwords.words('english'))
    trans_text = word_tokenize(text.lower())

    trans_text = [x for x in trans_text if not x in stop_words] # remove stopword
    trans_text = [x for x in trans_text if x.isalnum()]# remove punctations and special character

    # stemming
    stemming = PorterStemmer()
    trans_text = [stemming.stem(x) for x in trans_text]

    return " ".join(trans_text)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email Spam Classifier')

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    # pre-process
    transformed_sms = text_transform(input_sms)

    # vectorize
    vector = tfidf.transform([transformed_sms])

    # predict
    result = model.predict(vector)[0]

    # display
    if result == 1:
        st.error("Spam")
    else:
        st.success("Not Spam")
