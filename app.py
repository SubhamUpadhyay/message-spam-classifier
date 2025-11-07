import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
import gdown
import os

# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords

ps = PorterStemmer()

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

# Download files from Google Drive if not present
@st.cache_resource
def load_models():
    # Google Drive file IDs
    vectorizer_id = '1YlCs_XzRL83x7O74_gu1rr3ZbWSmRMJh'
    model_id = '1S7qd7hFI2aTfGUczzWtMxbYA9AVeE_WQ'
    
    # Download vectorizer if not exists
    if not os.path.exists('vectorizer.pkl'):
        vectorizer_url = f'https://drive.google.com/uc?id={vectorizer_id}'
        gdown.download(vectorizer_url, 'vectorizer.pkl', quiet=False)
    
    # Download model if not exists
    if not os.path.exists('model.pkl'):
        model_url = f'https://drive.google.com/uc?id={model_id}'
        gdown.download(model_url, 'model.pkl', quiet=False)
    
    # Load the models
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    
    return tfidf, model

tfidf, model = load_models()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")