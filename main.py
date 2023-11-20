import streamlit as st 
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("KernAI/stock-news-distilbert")

tokenizer = AutoTokenizer.from_pretrained("KernAI/stock-news-distilbert")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
sent = SentimentIntensityAnalyzer()


load_dotenv()

api_key = os.environ['API_KEY'] #api_key for marketaux

def mainSentiment():
    sentiment_label = [] #stores the sentiment label for each news head-line
    sentiment_score = []
    
    data = requests.get("https://api.marketaux.com/v1/news/all?filter_entities=true&language=en&countries=in&api_token="+api_key).json()
    #print(data)
    for i in range(0,len(data['data'])):
        print(data['data'][i]['title'])
        label = classifier(data['data'][i]['title'])[0]['label']
        sentiment_label.append(label)
        for j in range(0,len(data['data'][i]["similar"])):
            print(data['data'][i]["similar"][j]["title"])
            label = classifier(data['data'][i]["similar"][j]["title"])[0]['label']
            sentiment_label.append(label)
        
    
    st.text(sentiment_label)
        

st.set_page_config(page_title="StockPulse.AI📈🤖")
st.title("StockPulse.AI📈🤖")
st.text("Data app to analyze the sentiment of Indian Stock market!")

print(mainSentiment())
#st.text_are(mainSentiment())




