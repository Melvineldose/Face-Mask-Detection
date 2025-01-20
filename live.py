import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os

class MaskDetector:
    def __init__(self, model_path='face_mask_detection_model.h5'):
        # Load the face detection cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load the trained mask detection model
        self.model = load_model(model_path)
        
        # Define the classes
        self.classes = ['Without Mask', 'With Mask']
        
        # Define color for bounding boxes (BGR format)
        self.colors = {
            'With Mask': (0, 255, 0),      # Green
            'Without Mask': (0, 0, 255)     # Red
        }

    def preprocess_face(self, face_img):
        # Resize to the input shape of your model
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def detect_and_predict(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Preprocess face
            processed_face = self.preprocess_face(face_roi)
            
            # Make prediction
            prediction = self.model.predict(processed_face)
            label = self.classes[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            color = self.colors[label]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f'{label} ({confidence*100:.2f}%)'
            cv2.putText(frame, label_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        return frame

    def start_detection(self):
        cap = cv2.VideoCapture(0)
        
        print("Starting live detection... Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from camera")
                break
            output_frame = self.detect_and_predict(frame)
            
           
            cv2.imshow('Face Mask Detection', output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

def main():
    try:
        detector = MaskDetector()
        detector.start_detection()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()