import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from googletrans import Translator
from gtts import gTTS
import os
from collections import deque
import time
from datetime import datetime
from pydub import AudioSegment
import wave
import contextlib

# Initialize MediaPipe solutions
mp_holistic = mp.solutions.holistic

# Define the actions
actions = ['Good morning', 'How are you', 'I am fine', 'Thank you', 
           'What is your name', 'Nice to meet you', 'Can you help me', 'Listen to me']

# Prediction parameters
PREDICTION_THRESHOLD = 0.65  # Threshold for initial detection
CONFIDENCE_THRESHOLD = 0.8   # Threshold for confident predictions
MIN_CONSECUTIVE_FRAMES = 15  # Minimum frames for a valid prediction
PREDICTION_COOLDOWN = 1.5    # Cooldown time between predictions
MAX_PREDICTION_TIME = 3.0    # Maximum time to wait for confident prediction

class PredictionSmoother:
    def __init__(self, actions, window_size=30):
        self.actions = actions
        self.window_size = window_size
        self.prediction_counts = {action: 0 for action in actions}
        self.current_window = deque(maxlen=window_size)
        self.last_prediction = None
        self.last_prediction_time = time.time()
        self.current_sequence_start = time.time()
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        
    def update(self, prediction, confidence):
        current_time = time.time()
        
        # Reset if too much time has passed
        if current_time - self.current_sequence_start > MAX_PREDICTION_TIME:
            self.reset()
            self.current_sequence_start = current_time
            
        # Add new prediction to window
        if confidence > PREDICTION_THRESHOLD:
            self.current_window.append(prediction)
            
        # Count predictions in current window
        if len(self.current_window) >= MIN_CONSECUTIVE_FRAMES:
            counts = {}
            for pred in self.current_window:
                counts[pred] = counts.get(pred, 0) + 1
            
            # Find most common prediction
            if counts:
                most_common = max(counts.items(), key=lambda x: x[1])
                pred_action = most_common[0]
                pred_confidence = most_common[1] / len(self.current_window)
                
                # Check if prediction is confident and enough time has passed
                if (pred_confidence > self.confidence_threshold and 
                    current_time - self.last_prediction_time > PREDICTION_COOLDOWN):
                    self.last_prediction = pred_action
                    self.last_prediction_time = current_time
                    self.prediction_counts[pred_action] += 1
                    self.reset()
                    return pred_action
                    
        return None
    
    def reset(self):
        self.current_window.clear()
        self.current_sequence_start = time.time()
        
    def get_prediction_stats(self):
        return self.prediction_counts

class SignPredictor:
    def __init__(self, model_path='sign_model_v3.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.translator = Translator()
        self.sequence = []
        self.predictions = []
        self.current_sentence = []
        self.prediction_smoother = PredictionSmoother(actions)
        self.audio_dir = 'predicted_audio'
        os.makedirs(self.audio_dir, exist_ok=True)
        
    def mediapipe_detection(self, image, holistic):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def draw_landmarks(self, image, results):
        mp_drawing = mp.solutions.drawing_utils
        
        # Draw face connections
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        # Draw pose connections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        # Draw hand connections
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

    def create_audio(self):
        if not self.predictions:
            print("No predictions to process")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_audio_path = os.path.join(self.audio_dir, f'combined_predictions_{timestamp}.mp3')
        
        # Combine all predictions into one string with spaces
        full_text = ' '.join(self.predictions)
        
        try:
            # Translate entire text at once
            result = self.translator.translate(full_text, src='en', dest='ur')
            urdu_text = result.text
            
            print(f"\nFinal Translation:")
            print(f"English: {full_text}")
            print(f"Urdu: {urdu_text}")
            
            # Create single audio file
            tts = gTTS(text=urdu_text, lang='ur')
            tts.save(final_audio_path)
            
            print(f"\nSaved combined audio to: {final_audio_path}")
            return final_audio_path
            
        except Exception as e:
            print(f"Error creating audio: {e}")
            return None

    def run_detection(self):
        cap = cv2.VideoCapture(0)
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                frame = cv2.resize(frame, (640, 480))
                image, results = self.mediapipe_detection(frame, holistic)
                self.draw_landmarks(image, results)
                
                # Extract keypoints and make prediction
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-40:]  # Keep last 40 frames

                if len(self.sequence) == 40:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    predicted_idx = np.argmax(res)
                    confidence = res[predicted_idx]
                    
                    # Use smoother to get stable prediction
                    smooth_prediction = self.prediction_smoother.update(actions[predicted_idx], confidence)
                    
                    if smooth_prediction and (not self.current_sentence or smooth_prediction != self.current_sentence[-1]):
                        self.current_sentence.append(smooth_prediction)
                        if smooth_prediction not in self.predictions:
                            self.predictions.append(smooth_prediction)
                            print(f"New prediction added: {smooth_prediction}")

                # Display current sentence
                cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, ' | '.join(self.current_sentence[-5:]), (3, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow('Sign Language Detection', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        
        # Create final audio after detection ends
        print("\nPredicted sentences:", self.predictions)
        return self.create_audio()

# Main execution
if __name__ == "__main__":
    predictor = SignPredictor()
    audio_file = predictor.run_detection()
    
    if audio_file:
        print(f"\nProcessing complete! Audio saved to: {audio_file}")
        print("Predictions:", predictor.predictions)
    else:
        print("No predictions were recorded or there was an error creating the audio.")