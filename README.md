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
  
## Motivation
**What is sentiment analysis?**  
Sentiment analysis is simply working out if a piece of text is positive, neutral, or negative. An example of a happy tweet might be: **It's such beautiful day out! So happy to be hiking with friends** where a sad tweet might be **Had to put my dog down today :( RIP best friend**
  
  
**Why Twitter?**  
I chose to get my data from twitter
## Collecting Data

## Data Cleaning

### For Natural Language Processing
In order to get my data ready for any type of exploratory analysis or modeling I had
to: Remove any links, user '@' handles, and hashtags from the tweets.

To get sentiment, I used a package built into nltk called Vader or Valence Aware Dictionary and sEntiment Reasoner. It is a lexicon and rule-based sentiment analyzer that performs extremely well with sentiments expressed in social media. Social media is hard for natural language processing because people rarely use proper sentence structure and tend to use a lot of slang.
The majority of sentiment analysis approaches take one of two forms: polarity-based, where pieces of texts are classified as either positive or negative, or valence-based, where the intensity of the sentiment is taken into account