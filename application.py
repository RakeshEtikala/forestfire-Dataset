import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Import Ridge Regressor and Standard Scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            # Extract data from the form
            Temperature = float(request.form.get('Temperature'))
            RH = float(request.form.get('RH'))
            Ws = float(request.form.get('Ws'))
            Rain = float(request.form.get('Rain'))
            FFMC = float(request.form.get('FFMC'))
            DMC = float(request.form.get('DMC'))
            ISI = float(request.form.get('ISI'))
            Classes = float(request.form.get('Classes'))
            Region = float(request.form.get('Region'))

            # Scale the input data
            new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])

            # Make prediction
            result = ridge_model.predict(new_data_scaled)

            # Render template with the prediction result
            return render_template('home.html', result=result[0])

        except Exception as e:
            # If there's an error, render the home page with an error message
            return render_template('home.html', error=str(e))

    # Handle GET request by rendering the input form
    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
