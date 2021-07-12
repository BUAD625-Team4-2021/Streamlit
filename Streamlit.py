#!/usr/bin/env python
# coding: utf-8

# In[1]:
import streamlit as st
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


def checkairline(Tweet):
    if "@AmericanAir" in Tweet:
        airlinecheck = "AmericanAir"
    if "@SouthwestAir" in Tweet:
        airlinecheck = "SouthwestAir"
    if "@USAirways" in Tweet:
        airlinecheck = "USAirways"
    if "@united" in Tweet:
        airlinecheck = "united"
    if "@VirginAmerica" in Tweet:
        airlinecheck = "VirginAmerica"
    return airlinecheck





# In[3]:




# In[ ]:
st.title('Airline Sentiment Analysis')
st.subheader("Enter a Tweet in order to receive the aspect and sentiment")

#user_df = [["test"],["user"]]

def get_data():
    return []

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

    

test_data = pd.DataFrame(get_data(), columns =['Tweet', 'Airline', 'Sentiment','Aspect'], dtype = float)
st.write(test_data.iloc[-1])


fig = plt.figure()
plot1 = sns.displot(test_data, x='Airline', hue='Sentiment', multiple='stack')
st.pyplot(fig=plot1)

option = st.selectbox(
    'Filter Data to Selected Airline',
    ('AmericanAir', 'SouthwestAir', 'USAirways', 'united', 'VirginAmerica' ))


plot2_data = test_data.loc[(test_data['Airline'] == option)]

st.subheader("Aspect Analysis for "+ option)
fig = plt.figure()
plot2 = sns.displot(plot2_data, x='Aspect', hue='Sentiment', multiple='stack')
st.pyplot(fig=plot2)

st.subheader("Full List of Tweets")
st.dataframe(test_data,height = 500)
