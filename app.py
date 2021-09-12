import pickle
from flask import Flask, render_template, request
from engineering import TextParsing


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict/", methods=["POST"])
def predict():
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vocabulary.pkl", "rb"))
    message = request.form["OriginalTweet"]
    message = TextParsing(message).regexp()
    message = TextParsing(message).tokenization()
    data = [message]
    vector = vectorizer.transform(data).toarray()
    my_prediction = model.predict(vector)
    return render_template("result.html", prediction=my_prediction)


if __name__ == "__main__":
    app.run(port=5000, debug=True)

