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

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y_one_hot, test_size=0.1, random_state=8)

# Build the model
model = Sequential([
    Dense(100, input_shape=(40,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(200),
    Activation('relu'),
    Dropout(0.5),
    Dense(100),
    Activation('relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_)),
    Activation('softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath="C:/Users/renny/Desktop/Bird Sound/saved_models/audio_classification.hdf5", 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, 
          validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)
y_pred=model.predict(x_test)


duration = datetime.now() - start
print("Training completed in time: ", duration)

# Prediction
audio_path = "C:/Users/renny/Desktop/Bird Sound/Audios/Asian Koel 3.wav"
prediction_feature = features_extractor(audio_path)
prediction_feature = prediction_feature.reshape(1, -1)

predicted_probabilities = model.predict(prediction_feature)
predicted_label = np.argmax(predicted_probabilities)
predicted_class = label_encoder.inverse_transform([predicted_label])[0]

print(f"The predicted bird species is: {predicted_class}")