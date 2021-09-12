import re
import string
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


class TextParsing:

    def __init__(self, text):
        self.text = text

    def regexp(self):
        self.text = self.text.lower()
        self.text = re.sub(r"http\S+", "", self.text)
        self.text = re.sub("[%s]" % re.escape(string.punctuation), " ", self.text)
        return self.text

    def tokenization(self):
        tokens = word_tokenize(self.text)
        token_words = " ".join([wnl.lemmatize(word) for word in tokens if word not in english_stopwords])
        return token_words


corona = pd.read_csv("C:/Coding Repository/covid-19/Corona_NLP_train.csv", encoding='ISO-8859-1')

english_stopwords = stopwords.words("english")
wnl = WordNetLemmatizer()

corona["OriginalTweet"] = corona["OriginalTweet"].apply(lambda x: TextParsing(x).regexp())
corona["OriginalTweet"] = corona["OriginalTweet"].apply(lambda x: TextParsing(x).tokenization())

corona["Sentiment_integer"] = corona["Sentiment"].map({"Extremely Negative": 0,
                                                       "Negative": 1,
                                                       "Neutral": 2,
                                                       "Positive": 3,
                                                       "Extremely Positive": 4})

corona.drop(["Location", "UserName", "ScreenName", "TweetAt"], axis=1, inplace=True)

corona.to_csv("app_data/Corona_NLP_train_cleaned.csv", header=True, index=False)

