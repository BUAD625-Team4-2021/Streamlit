#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json


# In[54]:


st.title('Airline Sentiment Analysis')


# In[71]:


df = pd.read_csv("C:/Users/Lacey/Documents/Ryan/Udel/Tweets.csv")


# In[72]:


df.head()


# In[93]:


st.subheader('Total Tweets for Each Airline')


# In[95]:


fig, ax = plt.subplots()
df.airline.value_counts().plot(kind='barh')
st.pyplot(fig)


# In[98]:


st.subheader('Complaint Frequency')


# In[97]:


fig, ax = plt.subplots()
df.negativereason.value_counts().plot(kind='barh',figsize=(7,7))
plt.xlabel('Frequency')
plt.ylabel('Negative Reasons')
st.pyplot(fig)


# In[55]:


st.subheader("Live Tweets from Twitter")


# In[56]:


ACCESS_TOKEN = "1406225956444987396-UIQnOCMLD65Tc4eAREek0YwbPz4MoG"
ACCESS_TOKEN_SECRET = "4SLcS9EVEAKPGRf7GcKrQCFWqLypl6MwuHwV1s7ztUOuO"
CONSUMER_KEY = "gISINc0R1EYSwhmJfP37NC8Id"
CONSUMER_SECRET = "biD9WpUCVdrr1eMyQtfOrsroSU7TjLcEKAYPsKoPGb8laJ3wkg"


# In[63]:


# # # # TWITTER STREAMER # # # #
class TwitterStreamer():
    """
    Class for streaming and processing live tweets.
    """
    def __init__(self):
        pass

    def stream_tweets(self, fetched_tweets_filename, hash_tag_list):
        # This handles Twitter authetification and the connection to Twitter Streaming API
        listener = StdOutListener(fetched_tweets_filename)
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        stream = Stream(auth, listener)

        # This line filter Twitter Streams to capture data by the keywords: 
        stream.filter(track=hash_tag_list, languages=["en"])


# # # # TWITTER STREAM LISTENER # # # #
class StdOutListener(StreamListener):
    """
    This is a basic listener that just prints received tweets to stdout.
    """
    def __init__(self, fetched_tweets_filename):
        self.fetched_tweets_filename = fetched_tweets_filename

    def on_data(self, data):
        try:
            json_load = json.loads(data)
            text = {'text': json_load['text']}
            st.text(json.dumps(text))
            with open(self.fetched_tweets_filename, 'a') as tf:
                tf.write(data)
            return True
        except BaseException as e:
            print("Error on_data %s" % str(e))
        return True
          

    def on_error(self, status):
        print(status)


# In[64]:


if __name__ == '__main__':
 
    # Authenticate using config.py and connect to Twitter Streaming API.
    hash_tag_list = ["@AmericanAir", "@SouthwestAir", "@USAirways","@united", "@VirginAmerica"]
    fetched_tweets_filename = "tweets.txt"

    twitter_streamer = TwitterStreamer()
    twitter_streamer.stream_tweets(fetched_tweets_filename, hash_tag_list)


# In[ ]:





# In[ ]:




