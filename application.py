from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the model
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method == 'POST':
        Temprature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled  = standard_scaler.transform([[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        #data should be in the same order as the model was trained on
        #data should be given in a 2D array format

        result = ridge_model.predict(new_data_scaled)
        #this will return a 1D array with the prediction thats why we will use result[0] to get the first element

        return render_template('home.html', results = result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

