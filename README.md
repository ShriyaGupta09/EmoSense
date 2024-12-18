# Emotion Detector

This project is designed to detect emotions from facial expressions using a webcam. Follow the steps below to set up and run the project in Jupyter Notebook.

## Prerequisites

Before starting, make sure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Webcam (for emotion detection)

## Steps to Setup

### 1. Download the Dataset

You can download the dataset used for training the emotion detector model. The dataset contains images labeled with different emotions.

#### Download from Kaggle:

1. Go to the [Kaggle Facial Expression Recognition Dataset](https://www.kaggle.com/datasets/msambare/fer2013) page.
2. Log in to your Kaggle account (or create one if you don't have one).
3. Click on the "Download" button to get the dataset as a ZIP file.

#### Extract Dataset:

After downloading, extract the ZIP file to a folder, e.g., `datasets/fer2013/`.

### 2. Install Required Libraries

Open your terminal or Jupyter Notebook and install the required dependencies.

```bash
pip install -r requirements.txt
```

Or you can install them individually using:

```bash
pip install opencv-python
pip install tensorflow
pip install keras
pip install numpy
pip install matplotlib
```

### 3. Create a Jupyter Notebook

1. Open Jupyter Notebook by running `jupyter notebook` in your terminal.
2. Create a new notebook and name it `EmotionDetector.ipynb`.
3. Copy the code for the emotion detector into the notebook, or import the necessary functions.

### 4. Load the Pre-trained Model

The model used to detect emotions can be pre-trained. Load the pre-trained model using the code below:

```python
from keras.models import load_model
model = load_model('path_to_your_trained_model.h5')
```

### 5. Access the Webcam

You can use OpenCV to open the webcam and begin the emotion detection process.

```python
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Code to process frame, detect emotions, and display result
    # Example: emotion = detect_emotion(frame)
    
    # Display the resulting frame
    cv2.imshow('Emotion Detector', frame)
    
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close window
cap.release()
cv2.destroyAllWindows()
```

### 6. Run the Notebook

Once everything is set up, run the cells in your Jupyter Notebook to initialize the webcam and begin emotion detection.

### 7. Stop Webcam

Press `q` while the webcam window is open to stop the webcam feed.

## Conclusion

You have successfully set up the Emotion Detector project in Jupyter Notebook. The model should now be able to detect emotions from faces captured via the webcam.

---
