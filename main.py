import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
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

#end-calculation of sentiment
def calulateSentiment(sLabel,sScore):
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
    st.text("Average Sentiment Of market: "+str(round(np.average(sScore),2)*100)+"%")
    st.text(sCount)

    size_of_groups=[pcount,necount,nucount]
    labels=['Positive','Negative','Neutral']
    colors = ['green','red','gray']
    plt.pie(size_of_groups,colors=colors,labels=labels)
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    st.pyplot(p)
    
    
    





def mainSentiment():
    sentiment_label = [] #stores the sentiment label for each news head-line
    sentiment_score = []
    
    data = requests.get("https://api.marketaux.com/v1/news/all?filter_entities=true&language=en&countries=in&api_token="+api_key).json()
    #print(data)
    for i in range(0,len(data['data'])):
        label = classifier(data['data'][i]['title'])[0]['label']
        sentiment_label.append(label)
        sentiment_score.append(round(sent.polarity_scores(data['data'][i]['title'])['compound'],2))
        for j in range(0,len(data['data'][i]["similar"])):
            #print(data['data'][i]["similar"][j]["title"])
            label = classifier(data['data'][i]["similar"][j]["title"])[0]['label']
            sentiment_label.append(label)
            sentiment_score.append(round(sent.polarity_scores(data['data'][i]["similar"][j]["title"])["compound"],2))

    #st.text(sentiment_label)
    #st.text(sentiment_score)
    calulateSentiment(sentiment_label,sentiment_score)
        

st.set_page_config(page_title="StockPulse.AI📈🤖")
st.title("StockPulse.AI📈🤖")
st.text("Data app to analyze the sentiment of Indian Stock market!")

print(mainSentiment())




