import streamlit as st 
import pandas as pd
import os
import requests
from dotenv import load_dotenv

import json

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("KernAI/stock-news-distilbert")

tokenizer = AutoTokenizer.from_pretrained("KernAI/stock-news-distilbert")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
#result = classifier("Global Digital MRO Market Size To Worth USD 3396 Million By 2032 | CAGR of 12.4%")


load_dotenv()

api_key = os.environ['API_KEY'] #api_key for marketaux

def mainSentiment():
    sentiment_label = []
    sentiment_score = []
    
    data = requests.get("https://api.marketaux.com/v1/news/all?filter_entities=true&language=en&countries=in&industries=&api_token="+api_key).json()
    print(type(data))
    for i in range(0,len(data['data'])-1):
        label = classifier(data['data'][i]['title'])[0]['label']
        sentiment_label.append(label)
    
    st.text(sentiment_label)
        

st.set_page_config(page_title="StockPulse.AI📈🤖")
st.title("StockPulse.AI📈🤖")
st.text("Data app to analyze the sentiment of Indian Stock market!")

print(mainSentiment())
#st.text_are(mainSentiment())




