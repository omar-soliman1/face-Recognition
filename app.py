import cv2
import numpy as np
from keras.models import load_model



emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Define the emotions for the output
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use OpenCV's Haar Cascade Classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the region of interest (ROI) from the grayscale image
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the ROI to 64x64 (required input shape for the model)
        roi_resized = cv2.resize(roi_gray, (64, 64)) / 255.0  # Normalize the pixel values
        
        # Reshape the image to match the model's expected input shape (1, 64, 64, 1)
        roi_reshaped = np.reshape(roi_resized, (1, 64, 64, 1))

        # Predict emotion
        prediction = emotion_model.predict(roi_reshaped)
        max_index = np.argmax(prediction[0])
        predicted_emotion = emotion_labels[max_index]

        # Display the predicted emotion
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
