import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("ufos.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_feature = [np.array(int_features)]
    prediction = model.predict(final_feature)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template("index.html", prediction_text = "most likely :{}".format(countries[output]))



if __name__ == "__main__":
    app.run(debug=True)