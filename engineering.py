import re
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


english_stopwords = stopwords.words("english")
wnl = WordNetLemmatizer()


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

