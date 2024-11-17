import sys
import cv2
import numpy as np
import os
import tensorflow as tf
import serial  
from datetime import datetime
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout

class AnimalDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(1)

        self.model = tf.keras.models.load_model(
            r'C:\Users\dipes\OneDrive\Desktop\Animals\animal_detection.h5'
        )

        self.class_names = ["cow", "ox", "pig"]
        self.image_save_path = r'C:\Users\dipes\OneDrive\Desktop\captured_images'
        os.makedirs(self.image_save_path, exist_ok=True)

        self.video_label = QLabel()
        self.capture_button = QPushButton("Capture Image")
        self.capture_button.clicked.connect(self.capture_image)
        self.alarm_button = QPushButton("Trigger Alarm")
        self.alarm_button.clicked.connect(self.trigger_buzzer)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.alarm_button)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.arduino = serial.Serial('COM6', 9600, timeout=1)

        self.current_label = None

    def update_frame(self):
        """Update the video feed and perform animal detection."""
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            qt_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_img))

            self.current_frame = frame 
            self.detect_animal(frame)

    def detect_animal(self, frame, threshold=0.95):
        """Perform animal detection on the given frame with confidence threshold."""
        img_array = cv2.resize(frame, (224, 224)) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        max_prob = np.max(predictions[0])  
        class_idx = np.argmax(predictions[0]) 
        self.current_label = self.class_names[class_idx]

        if max_prob < threshold:
            self.current_label = "No Animal Detected"
            print("No valid animal detected.")
        else:
            print(f"Detected: {self.current_label}")

        if self.current_label in ["cow", "ox", "pig"]:
            self.capture_image()  # Save the image
            self.trigger_buzzer()  

    def capture_image(self):
        """Capture and save the current frame with the predicted label."""
        if hasattr(self, 'current_frame'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.image_save_path}/animal_{timestamp}.jpg"

            labeled_frame = self.current_frame.copy()
            if self.current_label:
                cv2.putText(
                    labeled_frame, self.current_label,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA
                )

            cv2.imwrite(filename, labeled_frame)
            print(f"Image saved as {filename}")

    def trigger_buzzer(self):
        """Send a signal to activate the buzzer via Arduino for 3 seconds."""
        try:
            self.arduino.write(b'1')  
            print("Buzzer signal sent!")
        except Exception as e:
            print(f"Error sending signal: {e}")

    def closeEvent(self, event):
        """Handle window close event to release resources."""
        self.cap.release()
        self.arduino.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnimalDetectionApp()
    window.setWindowTitle("Animal Detection with Arduino Buzzer")
    window.show()
    sys.exit(app.exec())