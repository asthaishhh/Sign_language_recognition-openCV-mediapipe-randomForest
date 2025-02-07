import cv2
import mediapipe as mp
import os
import time
import numpy as np

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Directory to save processed hand ROIs
save_folder = "processed_hand_data"
os.makedirs(save_folder, exist_ok=True)

# Parameters
image_size = 128  # Desired output size for all images
counter = 0  # Counter for saved images

# Start webcam
cap = cv2.VideoCapture(0)

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

            if hand_roi.size > 0:  # Ensure ROI is valid
                # Resize the cropped image to a fixed size
                resized_roi = cv2.resize(hand_roi, (image_size, image_size))

                # Normalize pixel values to [0, 1]
                normalized_roi = resized_roi / 255.0

                # Save processed ROI when 's' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    timestamp = time.time()
                    file_name = f"{save_folder}/Hand_{int(timestamp)}.jpg"
                    cv2.imwrite(file_name, (normalized_roi * 255).astype(np.uint8))  # Save as uint8
                    print(f"Saved: {file_name}")
                    counter += 1

                # Show the processed ROI
                cv2.imshow("Processed Hand ROI", resized_roi)

    # Display the main frame with landmarks
    cv2.imshow("Hand Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
