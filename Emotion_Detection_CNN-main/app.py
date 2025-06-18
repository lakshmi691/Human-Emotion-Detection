from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load models with corrected paths
face_classifier = cv2.CascadeClassifier(r'C:\Users\pavan\OneDrive\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\pavan\OneDrive\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\model.h5')
gender_classifier = load_model(r'C:\Users\pavan\OneDrive\Desktop\Emotion_Detection_CNN-main\Emotion_Detection_CNN-main\Gender_model.h5')

# Emotion and gender labels (removed Angry, Disgust, and Fear)
emotion_labels = ['Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

# Initialize emotion counts
emotion_counts = {label: 0 for label in emotion_labels}

def generate_frames():
    global emotion_counts  # Use the global variable to track counts
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces:
                # Draw rectangle around the face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Prepare the face for emotion and gender detection
                face = gray_frame[y:y + h, x:x + w]

                # Resize to the input size of the emotion model (48x48)
                emotion_face = cv2.resize(face, (48, 48))
                emotion_face = emotion_face.astype("float32") / 255.0  # Normalize
                emotion_face = img_to_array(emotion_face)
                emotion_face = np.expand_dims(emotion_face, axis=0)

                # Predict emotion
                emotion_prediction = emotion_classifier.predict(emotion_face)[0]  # Get prediction as 1D array
                emotion_prediction = emotion_prediction[-len(emotion_labels):]  # Use only the last labels
                emotion_label = emotion_labels[np.argmax(emotion_prediction)]

                # Update emotion counts
                emotion_counts[emotion_label] += 1

                # Resize to the input size of the gender model (256x256)
                gender_face = cv2.resize(frame[y:y + h, x:x + w], (256, 256))  # Use the original frame for RGB
                gender_face = gender_face.astype("float32") / 255.0  # Normalize
                gender_face = img_to_array(gender_face)
                gender_face = np.expand_dims(gender_face, axis=0)

                # Predict gender
                gender_prediction = gender_classifier.predict(gender_face)
                gender_label = gender_labels[np.argmax(gender_prediction)]

                # Display the emotion and gender on the frame
                cv2.putText(frame, f'Emotion: {emotion_label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f'Gender: {gender_label}', (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_counts', methods=['GET'])
def emotion_counts_data():
    return jsonify(emotion_counts)

if __name__ == '__main__':
    app.run(debug=True)
