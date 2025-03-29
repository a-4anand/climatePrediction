from flask import Flask, request, jsonify, render_template
import numpy as np
from keras.models import load_model
import pickle
from keras import losses
import pandas as pd

app = Flask(__name__)

# Load CNN model
model = load_model('weather_cnn_model.h5', custom_objects={'mse': losses.mean_squared_error})

# Load scaler for normalization (if used)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Prepare input based on lat, long (feature transformation for CNN)
def prepare_input(lat, lng):
    # Create input feature vector with correct feature names
    input_data = pd.DataFrame([[lng, lat]], columns=['LONGITUDE_mean', 'LATITUDE_mean'])

    # Scale input using StandardScaler
    input_data_scaled = scaler.transform(input_data)

    # Reshape input for CNN: (batch_size, time_steps, features)
    input_data_reshaped = input_data_scaled.reshape(1, input_data_scaled.shape[1], 1)

    return input_data_reshaped
# Route to render the map
@app.route('/')
def index():
    return render_template('map.html')

# Route to predict temperature using the CNN model
@app.route('/predict_temp', methods=['POST'])
def predict_temp():
    try:
        data_request = request.get_json()
        lat = float(data_request['lat'])
        lng = float(data_request['lng'])

        # Prepare input for the CNN model
        input_data = prepare_input(lat, lng)

        # Predict using CNN model
        predictions = model.predict(input_data)

        # Extract predictions for multiple days
        result = {
            'TMP_mean_day1': round(float(predictions[0][0]), 2),
            'TMP_max_day1': round(float(predictions[0][1]), 2),
            'TMP_min_day1': round(float(predictions[0][2]), 2)
        }
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
