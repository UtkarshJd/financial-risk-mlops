# deployment/app.py
from flask import Flask, request, jsonify
import joblib
import boto3
import pandas as pd
import os

app = Flask(__name__)

# Load model from S3 on startup
def load_model():
    s3 = boto3.client('s3')
    bucket = 'mlopsproject3'  # Your S3 bucket name
    model_key = 'models/risk_model.pkl'  # Path to your model in S3
    local_model_path = '/tmp/model.pkl'  # Temporary path in EB
    
    # Download model from S3
    s3.download_file(bucket, model_key, local_model_path)
    return joblib.load(local_model_path)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)