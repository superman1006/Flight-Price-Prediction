from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='../Web')
CORS(app)  # Enable CORS

# Load the model and the preprocessor
model = joblib.load(os.path.join(os.path.dirname(__file__), 'flight_price_model.pkl'))
preprocessor = joblib.load(os.path.join(os.path.dirname(__file__), 'preprocessor.pkl'))

# Define the expected columns
expected_columns = [
    'duration', 'days_left', 'airline_AirAsia', 'airline_Air_India', 'airline_GO_FIRST',
    'airline_Indigo', 'airline_SpiceJet', 'airline_Vistara', 'source_city_Bangalore',
    'source_city_Chennai', 'source_city_Delhi', 'source_city_Hyderabad', 'source_city_Kolkata',
    'source_city_Mumbai', 'departure_time_Afternoon', 'departure_time_Early_Morning',
    'departure_time_Evening', 'departure_time_Late_Night', 'departure_time_Morning',
    'departure_time_Night', 'stops_one', 'stops_two_or_more', 'stops_zero',
    'arrival_time_Afternoon', 'arrival_time_Early_Morning', 'arrival_time_Evening',
    'arrival_time_Late_Night', 'arrival_time_Morning', 'arrival_time_Night',
    'destination_city_Bangalore', 'destination_city_Chennai', 'destination_city_Delhi',
    'destination_city_Hyderabad', 'destination_city_Kolkata', 'destination_city_Mumbai',
    'class_Business', 'class_Economy'
]

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.static_folder, 'favicon.ico')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    # Add missing columns with default values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Preprocess the input data
    df_preprocessed = preprocessor.transform(df)

    # Make prediction
    prediction = model.predict(df_preprocessed)
    return jsonify({'price': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)