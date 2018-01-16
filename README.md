# Twitter Sentiment Analysis  
  
Tyler Atkinson
  
---
Using machine learning and natural-language-processing to determine whether someone is happy or not, based off of their tweets.

# Table of Contents
  1. [Motivation for project](#motivation)
  2. [Collecting data](#data)
  3. [Data Cleaning](#cleaning)
    - For NLP
    - For location
  
# Motivation
  
# Collecting Data

# Data Cleaning

## For Natural Language Processing
In order to get my data ready for any type of exploratory analysis or modeling I had
to: Remove any links, user '@' handles, and hashtags from the tweets.

To get sentiment, I used a package built into nltk called Vader or Valence Aware Dictionary and sEntiment Reasoner. It is a lexicon and rule-based sentiment analyzer that performs extremely well with sentiments expressed in social media. Social media is hard for natural language processing because people rarely use proper sentence structure when using social media. 