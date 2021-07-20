#!/usr/bin/env python
# coding: utf-8

#Import packages
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

#Load Aspects dictionary
aspect_assignments = {'Customer Service':['contact', 'emailed', 'fix', 'staff', 'speak', 'talk', 'care', 'experience', 'rep', 'issue', 'thanks', 'hold', 'thank', 'appreciate', 'response', 'service', 'customer', 'phone', 'agent', 'email', 'speak', 'help', 'please', 'call', 'refund', 'need'],
                      'Ongoing Flight(s)':['travel', 'wifi', 'leaving', 'updates', 'weather', 'attendant', 'connecting', 'early', 'arrived', 'landed', 'gate', 'delay', 'delayed', 'late', 'status', 'schedule', 'cancelled', 'cancel', 'pilots', 'pilot', 'passengers', 'passenger', 'boarding'],
                      'Booking': ['pass', 'credit', 'miles', 'hotel', 'app', 'fee', 'voucher', 'upgrade', 'class', 'available', 'website', 'online', 'book', 'booking', 'seats', 'seat', 'boarding', 'rebook', 'confirmation', 'reschedule', 'ticket', 'reserved'],
                      'Luggage': ['bag', 'check', 'lost', 'baggage', 'bags', 'luggage', 'claim'],
                      'Wait Times': ['wait', 'waited', 'stuck', 'line', 'hour', 'hours', 'minutes', 'days', 'today', 'tomorrow', 'time', 'min', 'hrs']
                      }
## Function to grab aspect based on word (key based on value)
@st.cache(allow_output_mutation=True)
def get_key(val):
    for key, value in aspect_assignments.items():
        for item in value:
            if (val == item):
                return key
            

# function to get the aspect for a tweet
@st.cache(allow_output_mutation=True)
def get_aspect_for_tweet(tweet):
    cs = 0
    of = 0
    bo = 0
    lu = 0
    wt = 0
    me = 0
    tweet_array = tweet.split(" ")
    for word in tweet_array:
        tof = False
        for value in aspect_assignments.values():
            if (word in value):
                tof = True
                aspect = get_key(word)
                if aspect == 'Customer Service':
                    cs += 1
                if aspect == 'Ongoing Flight(s)':
                    of += 1
                if aspect == 'Booking':
                    bo += 1
                if aspect == 'Luggage':
                    lu += 1
                if aspect == 'Wait Times':
                    wt += 1
                break

    assignments = {'cs': cs,'of': of,'bo': bo,'lu': lu,'me': me}
    test_value = max(assignments.values())
    test_key = 'me'
    if test_value > 0:
        for key, value in assignments.items():
            if test_value == value:
                test_key = key
                break
    aspects = {'cs': 'Customer Service', 'of': 'Ongoing Flight(s)', 'bo': 'Booking', 'lu': 'Luggage', 'me': "Miscellaneous"} 
    aspect = aspects[test_key]
    return aspect
            
#Load sentiment model
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


#Cache function to pick airline based on user input tweet
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

#Load in new tweet data from Github
new_data = pd.read_csv('https://github.com/BUAD625-Team4-2021/New_Tweets/blob/main/new_tweets_final_sentiment.csv?raw=true',index_col=[0])


#Create cache to hold all user input tweets
@st.cache(allow_output_mutation=True)
def get_data():
    return []


#Title and subtitle for streamlit
st.title('Airline Sentiment Analysis')
st.subheader("Enter a Tweet in order to receive the aspect and sentiment")

#Textbox for user input
user_input = st.text_input("Input Tweet")

#If button is pressed, analyze the tweet that has been entered into the text box
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

        final_score = labels[ranking[0]]
        airlinecheck = checkairline(user_input)
        aspect = get_aspect_for_tweet(text)
        
   
        
    get_data().append({"Tweet": user_input, "Airline": airlinecheck, "Sentiment": final_score, "Aspect": aspect})
    
#Combine data loaded from Github and new tweets entered by user
user_input_data = pd.DataFrame(get_data(), columns =['Tweet', 'Airline', 'Sentiment','Aspect'], dtype = float)
test_data = pd.concat([new_data,user_input_data]).reset_index(drop=True)

#Show full results from most recent tweet
st.write(test_data.iloc[-1])

#Colors for graphs
sns.set_palette("muted")
sns.set(font_scale=.75)

palette = {"positive":"#078112",
           "negative":"#AD3108", 
           "neutral":"#A59F9D"}

#Tweet distirbution graph
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

#Dropdown for Aspect Anlysis graph
option = st.selectbox(
    'Filter Data to Selected Airline',
    ('AmericanAir', 'SouthwestAir', 'Delta', 'united', 'JetBlue' ))

#Aspect Anlysis graph
plot2_data = test_data.loc[(test_data['Airline'] == option)]

palette = {"positive":"#078112",
           "negative":"#AD3108", 
           "neutral":"#A59F9D"}

sns.set(font_scale=.65)
fig = plt.figure()
plot2 = sns.displot(plot2_data, x='Aspect', hue='Sentiment', multiple='stack',palette = palette,hue_order=["positive","neutral","negative"])
plt.ylabel('Count of Tweets')
plt.xlabel('Aspect')
plt.title('Aspect Analysis for '+ option)
st.pyplot(fig=plot2)

#Print full tweet database
st.subheader("Full List of Tweets")
st.dataframe(test_data,height = 300)

        
    

