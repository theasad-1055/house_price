from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

data = pd.read_csv("output.csv")
pipe = pickle.load(open("ML_Regression.pkl", "rb"))


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        total_sqft = float(request.form.get('total_sqft'))

        # Basic validation
        if bhk <= 0 or bath <= 0 or total_sqft <= 0:
            return "Please enter values greater than 0."

        # Create DataFrame
        input_df = pd.DataFrame(
            [[location, bhk, bath, total_sqft]],
            columns=['location', 'bhk', 'bath', 'total_sqft']
        )

        prediction = pipe.predict(input_df)[0] * 1e5

        # If prediction is negative, make it 0
        if prediction < 0:
            prediction = 0

        # MESSAGE WHEN PREDICTION = 0
        if prediction == 0:
            return "According to your details, price could not be calculated for this place. Please update your BHK, Bathrooms or Square Feet and try again."

        return str(round(prediction, 2))

    except Exception as e:
        print("ERROR:", e)
        return "Error: Invalid input. Please check your values."


if __name__ == '__main__':
    app.run(debug=True, port=5001)
