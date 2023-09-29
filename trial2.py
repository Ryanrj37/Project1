import os
import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define paths
audio_dataset_path = "C:/Users/renny/Desktop/Bird Sound/Audios/"
metadata_path = "C:/Users/renny/Desktop/Bird Sound/Metadata/Birds.xlsx"

# Load metadata
metadata = pd.read_excel(metadata_path, sheet_name='Sheet1')
#"C:\Users\renny\Desktop\Bird Sound\Audios\Ashy Prinia 1.mp3"
# Define function to extract MFCC features
def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Extract features and labels
extracted_features = []
for index, row in metadata.iterrows():
    filename = os.path.join(audio_dataset_path, str(row["Filename"]))
    final_class_labels = row["Output"]
    data = features_extractor(filename)
    extracted_features.append([data, final_class_labels])

# Create DataFrame
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])

# Prepare data for model training
x = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the model
model = LogisticRegression()
model.fit(x_train, y_train)
ypred=model.predict(x_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
con=accuracy_score(y_test,ypred)
print(con)
import joblib
joblib.dump(model,'C:/Users/renny/Desktop/Bird Sound/saved_models/classifier.pkl')

# Prediction
audio_path = "C:/Users/renny/Desktop/Bird Sound/Audios/Red Junglefowl 2.wav"
prediction_feature = features_extractor(audio_path)
prediction_feature = prediction_feature.reshape(1, -1)

predicted_probabilities = model.predict(prediction_feature)


print(f"The predicted bird species is: {predicted_probabilities}")