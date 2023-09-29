import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Define paths
audio_dataset_path = "C:/Users/renny/Desktop/Bird Sound/Audios/"
metadata_path = "C:/Users/renny/Desktop/Bird Sound/Metadata/Birds.xlsx"

# Load metadata
metadata = pd.read_excel(metadata_path, sheet_name='Sheet1')

# Load the pre-trained model
import joblib
model = joblib.load("C:/Users/renny/Desktop/Bird Sound/saved_models/classifier.pkl")  # Replace with the actual path to your model file

# Define function to extract MFCC features
def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

@app.route('/classify-audio', methods=['POST'])
def classify_audio():
    try:
        # Get the uploaded audio file from the request
        audio_file = request.files['audioFile']
        if audio_file:
            # Save the uploaded file to a temporary location
            temp_audio_path = 'temp_audio.mp3'
            audio_file.save(temp_audio_path)

            # Extract features from the uploaded audio
            prediction_feature = features_extractor(temp_audio_path)
            prediction_feature = prediction_feature.reshape(1, -1)

            # Use the pre-trained model to classify the audio
            predicted_class = model.predict(prediction_feature)[0]

            # Return the classification result
            return jsonify({'result': predicted_class})
        else:
            return jsonify({'error': 'No audio file uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)