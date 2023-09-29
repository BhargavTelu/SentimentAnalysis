import pandas as pd
import streamlit as st
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer, BertForSequenceClassification
import torch

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def preprocess(x):
  x = x.lower()
  x = x.strip()
  x = re.sub('https\S+|www\S+', '', x) ## removing links
  x = x.split()
  x = [word for word in x if word not in stop_words]
  x = [ps.stem(word) for word in x]
  x = ' '.join(x)
  return x

model = BertForSequenceClassification.from_pretrained("Bhargav6239/FineTuneBert", use_auth_token = "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")

def classify_text(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(text,padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predictions = predictions.cpu().detach().numpy()
    if predictions[0,0]>=predictions[0,1]:
        return "Fake"
    else:
        return "Real" 
    
st.title('Fake News Classifier')
text = st.text_area('Enter some text')
button = st.button("Submit")
text = preprocess(text)

if text and button:
    with st.spinner('Classifying...'):
        label = classify_text(text)
        st.write(f'This news is {label} news.')