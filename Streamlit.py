#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
import altair as alt
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request



# Preprocess text (username and link placeholders)
@st.cache(allow_output_mutation=True)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)
# In[2]:

@st.cache(allow_output_mutation=True)
def checkairline(Tweet):
    if "@AmericanAir" in Tweet:
        airlinecheck = "AmericanAir"
    if "@SouthwestAir" in Tweet:
        airlinecheck = "SouthwestAir"
    if "@Delta" in Tweet:
        airlinecheck = "Delta"
    if "@united" in Tweet:
        airlinecheck = "united"
    if "@JetBlue" in Tweet:
        airlinecheck = "JetBlue"
    return airlinecheck

new_data = pd.read_csv('https://github.com/BUAD625-Team4-2021/New_Tweets/blob/main/new_tweets_final_sentiment.csv?raw=true',index_col=0)

# In[3]:

@st.cache(allow_output_mutation=True)
def get_data():
    return []

# In[ ]:
st.title('Airline Sentiment Analysis')
st.subheader("Enter a Tweet in order to receive the aspect and sentiment")

#user_df = [["test"],["user"]]


user_input = st.text_input("Input Tweet")


if st.button("Analyze Tweet"):
    for tweet in user_input:
    
        text = user_input
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            #print(f"{i+1}) {l} {np.round(float(s), 4)}")

        final_score = labels[ranking[0]]
        airlinecheck = checkairline(user_input)
   
        
    
    get_data().append({"Tweet": user_input, "Airline": airlinecheck, "Sentiment": final_score, "Aspect": "aspect"})
    #test_data.append({"Tweet": user_input, "Airline": airlinecheck, "Sentiment": final_score, "Aspect": "aspect"}, ignore_index=True)

user_input_data = pd.DataFrame(get_data(), columns =['Tweet', 'Airline', 'Sentiment','Aspect'], dtype = float)

test_data = pd.concat([new_data,user_input_data]).reset_index(drop=True)

st.write(test_data.iloc[-1])

sns.set_palette("muted")
sns.set(font_scale=.75)

palette = {"positive":"#078112",
           "negative":"#AD3108", 
           "neutral":"#A59F9D"}

fig = plt.figure()
plot1 = sns.catplot(x = "Airline",       # x variable name
            hue = "Sentiment",  # group variable name
            data = test_data,     # dataframe to plot
            kind = "count",
            palette=palette,
            hue_order = ["positive","neutral","negative"])
plt.ylabel('Count of Tweets')
plt.title('Tweet Distribution')
st.pyplot(fig=plot1)

option = st.selectbox(
    'Filter Data to Selected Airline',
    ('AmericanAir', 'SouthwestAir', 'Delta', 'united', 'JetBlue' ))


plot2_data = test_data.loc[(test_data['Airline'] == option)]

palette = {"positive":"#078112",
           "negative":"#AD3108", 
           "neutral":"#A59F9D"}

fig = plt.figure()
plot2 = sns.displot(plot2_data, x='Aspect', hue='Sentiment', multiple='stack',palette = palette,hue_order=["positive","neutral","negative"])
plt.ylabel('Count of Tweets')
plt.xlabel('Aspect')
plt.title('Aspect Analysis for '+ option)
st.pyplot(fig=plot2)


st.subheader("Full List of Tweets")
st.dataframe(test_data,height = 300)

        
    

