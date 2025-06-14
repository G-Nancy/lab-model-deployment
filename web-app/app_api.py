import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("ufos.pkl", 'rb'))


@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        seconds = int(request.get_json()['seconds'])
        latitude = int(request.get_json()['latitude'])
        longitude = int(request.get_json()['longitude'])
        # seconds = int(request.form['seconds'])
        # latitude = int(request.form['latitude'])
        # longitude = int(request.form['longitude'])

        parameters = [seconds,latitude, longitude ]

        output = model.predict([np.array(parameters)])[0]
        countries = ["Australia", "Canada", "Germany", "UK", "US"]

        return {'prediction_text':countries[output]}
    else:
        return {"status": "alive"}


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)