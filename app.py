# from flask import Flask, request, jsonify
# from keras.models import load_model
# import joblib
# import pandas as pd
# import numpy as np
# import logging

# app = Flask(__name__)

# # Konfigurasi logging
# logging.basicConfig(filename='api.log', level=logging.ERROR, 
#                     format='%(asctime)s %(levelname)s: %(message)s')

# # Load model dan scaler
# try:
#     classification_model = load_model('Model/classification_model.h5')
#     regression_model = load_model('Model/deep_learning_regression_model.h5')
#     classification_scaler = joblib.load('Model/scaler_classification.pkl')
#     regression_scaler = joblib.load('Model/scaler_deep_learning_regression.pkl')
#     print("Model dan scaler berhasil di-load.")
# except Exception as e:
#     logging.error(f"Error saat me-load model atau scaler: {e}")
#     raise

# def predict_rad(features, scaler, model):
#     try:
#         feature_df = pd.DataFrame([features])
#         scaled_features = scaler.transform(feature_df)
#         predicted_value = model.predict(scaled_features)
#         return predicted_value[0][0]
#     except Exception as e:
#         logging.error(f"Error dalam prediksi rad(m): {e}")
#         raise

# def predict_condition(input_features, scaler, model):
#     try:
#         new_data = np.array([[input_features['Tn'], input_features['Tx'], input_features['Tavg'], 
#                               input_features['RH_avg'], input_features['ff_avg'], input_features['rad_m']]])
#         new_data_scaled = scaler.transform(new_data)
#         new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))
#         prediction = model.predict(new_data_reshaped)
#         predicted_label = (prediction > 0.5).astype(int)
#         return 'Aman' if predicted_label[0][0] == 0 else 'Tidak Aman'
#     except Exception as e:
#         logging.error(f"Error dalam prediksi kondisi: {e}")
#         raise

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         input_data = request.json

#         # Validasi input
#         required_keys = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'ff_avg']
#         for key in required_keys:
#             if key not in input_data:
#                 return jsonify({"error": f"Field {key} is missing"}), 400

#         # Konversi data ke float
#         for key in input_data:
#             input_data[key] = float(input_data[key])

#         # Prediksi rad(m)
#         rad_value = predict_rad(input_data, regression_scaler, regression_model)
#         input_data['rad_m'] = rad_value

#         # Prediksi kondisi
#         condition = predict_condition(input_data, classification_scaler, classification_model)

#         return jsonify({
#             "rad_m": rad_value,
#             "condition": condition
#         })
#     except Exception as e:
#         logging.error(f"Error saat prediksi: {e}")
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080)


# import pandas as pd
# import numpy as np
# from keras.models import load_model
# import joblib
# from flask import Flask, request, jsonify
# import logging

# # Initialize Flask app
# app = Flask(__name__)

# logging.basicConfig(filename='api.log', level=logging.ERROR, 
#                     format='%(asctime)s %(levelname)s: %(message)s')

# # Load models and scalers
# try:
#     classification_model = load_model('Model/classification_model.h5')
#     classification_scaler = joblib.load('Model/scaler_classification.pkl')
#     regression_model = load_model('Model/deep_learning_regression_model.h5')
#     regression_scaler = joblib.load('Model/scaler_deep_learning_regression.pkl')
#     print("Model dan scaler berhasil di-load.")
# except Exception as e:
#     print(f"Error saat me-load model atau scaler: {e}")
#     raise  # Raise exception to stop the application

# # Function to predict rad(m)
# def predict_rad(features, scaler, model):
#     try:
#         feature_df = pd.DataFrame([features])
#         scaled_features = scaler.transform(feature_df)
#         predicted_value = model.predict(scaled_features)
#         return predicted_value[0][0]
#     except Exception as e:
#         print(f"Error dalam prediksi rad(m): {e}")
#         raise

# # Function to predict condition
# def predict_condition(input_features, scaler, model):
#     try:
#         new_data = np.array([[input_features['Tn'], input_features['Tx'], input_features['Tavg'], 
#                               input_features['RH_avg'], input_features['ff_avg'], input_features['rad_m']]])
#         new_data_scaled = scaler.transform(new_data)
#         new_data_reshaped = new_data_scaled.reshape((new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))
#         prediction = model.predict(new_data_reshaped)
#         predicted_label = (prediction > 0.5).astype(int)
#         return 'Aman' if predicted_label[0][0] == 0 else 'Tidak Aman'
#     except Exception as e:
#         print(f"Error dalam prediksi kondisi: {e}")
#         raise

# # API endpoint for predicting rad(m)
# @app.route('/predict_rad', methods=['POST'])
# def api_predict_rad():
#     try:
#         features = request.json
#         predicted_rad = predict_rad(features, regression_scaler, regression_model)
#         return jsonify({'predicted_rad': predicted_rad}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # API endpoint for predicting condition
# @app.route('/predict_condition', methods=['POST'])
# def api_predict_condition():
#     try:
#         input_features = request.json
#         predicted_condition = predict_condition(input_features, classification_scaler, classification_model)
#         return jsonify({'predicted_condition': predicted_condition}), 200
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Run the Flask app
# if __name__ == '__main__':
#     app.run(debug=True)

# app.py

from flask import Flask, request, jsonify
from model import predict_rad, predict_condition
from keras.models import load_model
import joblib
import logging

app = Flask(__name__)

# Konfigurasi logging
logging.basicConfig(filename='api.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load model dan scaler di sini
try:
    classification_model = load_model('Model/classification_model.h5')
    regression_model = load_model('Model/deep_learning_regression_model.h5')
    classification_scaler = joblib.load('Model/scaler_classification.pkl')
    regression_scaler = joblib.load('Model/scaler_deep_learning_regression.pkl')
    print("Model dan scaler berhasil di-load.")
except Exception as e:
    print(f"Error saat me-load model atau scaler: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Expecting JSON input

        # Convert input data to float
        for key in data:
            data[key] = float(data[key])

        predicted_rad_value = predict_rad(data, regression_scaler, regression_model)

        data['rad_m'] = predicted_rad_value

        predicted_condition_value = predict_condition(data, classification_scaler, classification_model)

        return jsonify({
            'predicted_condition': predicted_condition_value,
            'predicted_rad': predicted_rad_value,
            'input_data': data
        })

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({'error': 'Terjadi error saat melakukan prediksi.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)