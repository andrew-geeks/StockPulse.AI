import streamlit as st 
import pandas as pd
import os
import requests
from dotenv import load_dotenv
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer

nltk.download('vader_lexicon')
model = AutoModelForSequenceClassification.from_pretrained("KernAI/stock-news-distilbert")

tokenizer = AutoTokenizer.from_pretrained("KernAI/stock-news-distilbert")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
sent = SentimentIntensityAnalyzer()


load_dotenv()

api_key = os.environ['API_KEY'] #api_key for marketaux


def calulateSentiment(sLabel):
    pcount=0
    necount=0
    nucount=0
    for i in sLabel:
        if(i=='positive'):
            pcount+=1
        elif(i=='negative'):
            necount+=1
        else:
            nucount+=1
    
    sCount={'positive':pcount,'negative':necount,'neutral':nucount}
    st.text(sCount)
    
    





def mainSentiment():
    sentiment_label = [] #stores the sentiment label for each news head-line
    sentiment_score = []
    
    data = requests.get("https://api.marketaux.com/v1/news/all?filter_entities=true&language=en&countries=in&api_token="+api_key).json()
    #print(data)
    for i in range(0,len(data['data'])):
        #print(data['data'][i]['title'])
        label = classifier(data['data'][i]['title'])[0]['label']
        sentiment_label.append(label)
        sentiment_score.append(sent.polarity_scores(label))
        for j in range(0,len(data['data'][i]["similar"])):
            #print(data['data'][i]["similar"][j]["title"])
            label = classifier(data['data'][i]["similar"][j]["title"])[0]['label']
            sentiment_label.append(label)
            sentiment_score.append(round(sent.polarity_scores(data['data'][i]["similar"][j]["title"])["compound"],2))

    st.text(sentiment_label)
    calulateSentiment(sentiment_label)
    st.text(sentiment_score)
        

st.set_page_config(page_title="StockPulse.AI📈🤖")
st.title("StockPulse.AI📈🤖")
st.text("Data app to analyze the sentiment of Indian Stock market!")

print(mainSentiment())
#st.text_are(mainSentiment())




