import cv2
import mediapipe as mp
import os
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)



# Directory to save the hand ROIs
save_folder = "dataset/2"
os.makedirs(save_folder, exist_ok=True)

# List to store features and labels
feature_list = []
label_list = []

# Start webcam
cap = cv2.VideoCapture(0)
image_size = 128
counter = 0  # Counter for saved images

#Load existing data if files exist
if os.path.exists('features_list.npy') and os.path.exists('labels_list.npy'):
    features_list = list(np.load('features_list.npy', allow_pickle=True))
    labels_list = list(np.load('labels_list.npy', allow_pickle=True))
else:
    features_list = []
    labels_list = []

# if os.path.exists('features_list.npy') and os.path.exists('labels_list.npy'):
#      feature_list = list(np.load('features_list.npy', allow_pickle=True))
#      label_list = list(np.load('labels_list.npy', allow_pickle=True))
# else:
#      feature_list = []
#      label_list = []

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB as Mediapipe processes images in RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe Hands
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the 3D coordinates of the hand landmarks (21 landmarks, each with x, y, z)
            hand_features = []
            for lm in hand_landmarks.landmark:
                hand_features.extend([lm.x, lm.y, lm.z])  # Flattened 3D coordinates
            
             # Extract bounding box coordinates
            h, w, _ = frame.shape  # Frame dimensions
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Add padding to the bounding box
            offset = 40
            x_min = max(0, x_min - offset)
            y_min = max(0, y_min - offset)
            x_max = min(w, x_max + offset)
            y_max = min(h, y_max + offset)

            # Crop the hand region (ROI)
            hand_roi = frame[y_min:y_max, x_min:x_max]

            # Display the cropped ROI
            if hand_roi.size > 0:  # Ensure ROI is valid
                resized_roi=cv2.resize(hand_roi,(image_size,image_size))
                cv2.imshow("Hand ROI", resized_roi)

                # Save ROI when 's' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    timestamp = time.time()
                    file_name = f"{save_folder}/Hand_{counter}.jpg"
                    cv2.imwrite(file_name, resized_roi)
                    print(f"Saved: {file_name}")
                    counter += 1


                    # Here, you can assign a label based on the gesture
                    # For example: 0 = "Open Hand", 1 = "Fist"
                    label = 2  # Update this based on the gesture you want to classify

                    features_list.append(hand_features)
                    labels_list.append(label)

                    # #becauce i made a mistake of saving different features into one folder!
                    # feature_list.append(hand_features)
                    # label_list.append(label) 

    # Display the main frame with landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Convert features and labels into numpy arrays
features_array = np.array(features_list)
labels_array = np.array(labels_list)
# feature_array = np.array(feature_list)
# label_array = np.array(label_list)

# Save the features and labels to a file
np.save('features_list.npy', features_list)  # changes features_list to feature_list and labels_list to label_list every where in the code!
np.save('labels_list.npy', labels_list)
print("Features and labels saved!")

    # Release resources
cap.release()
cv2.destroyAllWindows()

# if __name__ == "__main__":
#     storingLandmarks()

