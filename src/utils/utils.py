from transformers import AutoTokenizer, BertForSequenceClassification, BertTokenizer
import re
from wordcloud import WordCloud #Word visualization
import matplotlib.pyplot as plt
import os 

def preprocess_text(text):

        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove TAGs
        text = re.sub(r'@\w+', '', text)

        # Remove abbreviations
        text = re.sub(r'\b\'ll\b', ' will', text)
        text = re.sub(r'\b\'s\b', ' is', text)
        text = re.sub(r'\b\'re\b', ' are', text)
        text = re.sub(r'\b\'d\b', ' would', text)
        text = re.sub(r'\b\'m\b', ' am', text)
        text = re.sub(r'\b\'ve\b', ' have', text)
        # Remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Replace more than one space with a single space
        text = re.sub(r'\s+', ' ', text)
        # Trim text with space before and after
        text = text.strip()
        # Convert to lowercase
        text = text.lower()

        return text

def preprocess_wrapper(args):
        index, row = args
        row['tweet_text_processed'] = preprocess_text(row['tweet_text'])
        return row


def get_word_cloud(df,logger,parameters):
        positive_tweets = df[df["sentiment_label"] == 4]['tweet_text_processed'].astype(str)
        negative_tweets = df[df["sentiment_label"] == 0]['tweet_text_processed'].astype(str)
        draw_word_cloud(positive_tweets, parameters, sentiment_type = 1)
        draw_word_cloud(negative_tweets, parameters, sentiment_type = 0)
        # Joining all positive tweets into a single string without spaces
        
def draw_word_cloud(tweets, parameters, sentiment_type):
        word_cloud_text = ' '.join(tweets)

        #Creation of wordcloud
        wordcloud = WordCloud(
        max_font_size=100,
        max_words=100,
        background_color="black",
        scale=10,
        width=800,
        height=800
        ).generate(word_cloud_text)
        #Figure properties
        plt.figure(figsize=(10,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        if sentiment_type:
                plt.savefig(os.path.join(parameters['result_path'], f'cloud_words_positive.png'))
        else:
                plt.savefig(os.path.join(parameters['result_path'], f'cloud_words_negative.png'))
