import cv2
import tensorflow as tf
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("emotiondetector.json", 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start the webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral'}

# OpenCV window settings
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("output", cv2.WND_PROP_TOPMOST, 1)

while True:
    i, im = webcam.read()
    if not i:
        print("Webcam feed not accessible.")
        break
    print("Webcam is working.")

    # Convert image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            # Crop the face from the image and preprocess it
            image = gray[q:q + s, p:p + r]
            cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            
            # Make prediction
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]

            # Display the prediction label on the image
            cv2.putText(im, '%s' % (prediction_label), (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display the result in a window
        cv2.imshow("output", im)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except cv2.error:
        pass

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()

