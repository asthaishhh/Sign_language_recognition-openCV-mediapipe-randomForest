# Sign_language_recognition-openCV-mediapipe-randomForest
Sign Language Recognition using OpenCV, Mediapipe, and RandomForest
Overview
This project implements Sign Language Recognition using OpenCV and Mediapipe for hand tracking and feature extraction. The extracted hand landmarks are then classified using a RandomForest model.

Technologies Used
Python
OpenCV (for image processing)
Mediapipe (for hand landmark detection)
RandomForest Classifier (for sign classification)
Project Structure
hand_tracking.py → Captures hand landmarks using Mediapipe
data_collection.py → Collects sign language data and saves it
model_training.py → Trains the RandomForest model
sign_prediction.py → Predicts sign language gestures in real-time

The dataset consists of hand landmark coordinates extracted using Mediapipe. Data is stored in a CSV file with corresponding labels for different signs.

Results & Future Improvements
Achieved accurate sign recognition using RandomForest.
Future improvements:
Train with a larger dataset
Implement deep learning for better accuracy
Add more sign language gestures
Contributors
Astha Patel (GitHub)
