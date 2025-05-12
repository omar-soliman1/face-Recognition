Face Recognition and Emotion Detection System

This is a Python-based project that integrates real-time face recognition and emotion detection using OpenCV, Haar Cascades, the LBPH face recognizer, and a deep learning model trained on the FER2013 dataset.

Features:

Real-time face detection and recognition using OpenCV and Haar Cascade classifiers
Captures and stores grayscale facial images for each user
Trains a face recognizer using LBPH (Local Binary Patterns Histograms)
Recognizes faces from a webcam feed and displays the matched name
Detects and classifies human emotions using a pre-trained deep learning model
Displays emotion labels such as Happy, Sad, Angry, etc., on the detected faces
Modular structure with separate Python scripts for each task

Modules:

data_collect.py - For capturing and saving face data
training.py - For training the face recognition model
recognizer.py - For real-time recognition and emotion detection

Requirements:

Python 3.6 or higher
OpenCV (opencv-python)
NumPy
Keras and TensorFlow (for emotion detection)
Standard Python libraries: os, shutil, etc.

How to Use:

1. Collect Facial Data
Run the following command to start capturing facial images for training:
python data_collect.py
Captured images will be saved inside the dataSet folder, organized by user ID.

2. Train the Face Recognition Model
After collecting data, train the recognizer model using:
python training.py
A trained model will be saved as recognizer/trainingData.yml.

3. Run Real-Time Face Recognition and Emotion Detection
Start the recognition system with:
python recognizer.py
The webcam feed will open.
Detected faces will be matched with trained data.
The recognized name and predicted emotion will be displayed on the video stream.

Notes:

-Ensure the webcam is connected and accessible before running the scripts

-The emotion detection model must be placed correctly and referenced in the script

-You can update the emotion model path or use another pre-trained model as needed

