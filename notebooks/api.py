import sys
import os

# 将项目根目录添加到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd


app = Flask(__name__,
            static_folder='../data',  # 设置静态文件夹为data目录
            template_folder='../Web'  # 设置模板文件夹为Web目录
            )
CORS(app)  # Enable CORS

# Load the complete pipeline
model = joblib.load(os.path.join(os.path.dirname(__file__), 'flight_price_model.pkl'))

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
    return render_template('home.html')


@app.route('/predict')
def predict():
    return send_from_directory(app.template_folder, 'index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.template_folder, 'favicon.ico')


@app.route('/api/predict', methods=['POST'])
def predict_price():
    try:
        data = request.json

        # Create a DataFrame with all expected columns initialized to 0
        df = pd.DataFrame(0, index=[0], columns=expected_columns)

        # Update numeric features
        df['duration'] = float(data.get('duration', 0))
        df['days_left'] = float(data.get('days_left', 0))

        # Update categorical features
        airline = data.get('airline', '')
        source_city = data.get('source_city', '')
        destination_city = data.get('destination_city', '')
        departure_time = data.get('departure_time', '')
        arrival_time = data.get('arrival_time', '')
        stops = data.get('stops', '0')
        flight_class = data.get('class', '')

        # Update airline
        if airline:
            df[f'airline_{airline}'] = 1

        # Update source city
        if source_city:
            df[f'source_city_{source_city}'] = 1

        # Update destination city
        if destination_city:
            df[f'destination_city_{destination_city}'] = 1

        # Update departure time
        if departure_time:
            df[f'departure_time_{departure_time}'] = 1

        # Update arrival time
        if arrival_time:
            df[f'arrival_time_{arrival_time}'] = 1

        # Update stops
        if stops == '0':
            df['stops_zero'] = 1
        elif stops == '1':
            df['stops_one'] = 1
        elif stops == '2':
            df['stops_two_or_more'] = 1

        # Update class
        if flight_class:
            df[f'class_{flight_class}'] = 1

        # Make prediction using the complete pipeline
        prediction = model.predict(df)

        return jsonify({
            'price': float(prediction[0]),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400


if __name__ == '__main__':
    app.run(debug=True)
