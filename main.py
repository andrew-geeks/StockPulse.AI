import streamlit as st 
import pandas as pd
import os
from dotenv import load_dotenv
import pickle

load_dotenv()

api_key = os.environ['API_KEY'] #api_key for marketaux

st.set_page_config(page_title="StockPulse.AI📈🤖")
st.title("StockPulse.AI📈🤖")
st.text("Data app to analyze the sentiment of Indian Stock market!")



