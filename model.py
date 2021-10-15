import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from engineering import TextParsing

# Read initial csv
corona = pd.read_csv("C:/Coding Repository/covid-19-tweets-nlp/Corona_NLP_train.csv", encoding='ISO-8859-1')

# Pre-processing
corona["OriginalTweet"] = corona["OriginalTweet"].apply(lambda x: TextParsing(x).regexp())
corona["OriginalTweet"] = corona["OriginalTweet"].apply(lambda x: TextParsing(x).tokenization())

corona["Sentiment_integer"] = corona["Sentiment"].map({"Extremely Negative": 0,
                                                       "Negative": 1,
                                                       "Neutral": 2,
                                                       "Positive": 3,
                                                       "Extremely Positive": 4})

corona.drop(["Location", "UserName", "ScreenName", "TweetAt"], axis=1, inplace=True)

# Create new csv file
corona.to_csv("app_data/Corona_NLP_train_cleaned.csv", header=True, index=False)

# Read csv
corona = pd.read_csv("C:/Coding Repository/covid-19-tweets-nlp/app_data/Corona_NLP_train_cleaned.csv",
                     encoding='ISO-8859-1')
corona = corona.dropna()

# Consider train and target data
X = corona["OriginalTweet"]
y = corona["Sentiment_integer"]

# Save the vocabulary
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
pickle.dump(vectorizer, open("vocabulary.pkl", "wb"))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Save the model
model = SVC(kernel="linear", gamma="auto")
model.fit(X_train, y_train)
pickle.dump(model, open("model.pkl", "wb"))

