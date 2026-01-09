"""
video_capture.py
Handles webcam feed acquisition and blending with black background.
"""
import cv2
import numpy as np

class VideoCapture:
    def __init__(self, width=1280, height=720):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (self.width, self.height))
        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def blend_with_black(self, frame, alpha=0.25):
        """
        Softly blends the webcam frame over a pure black background.
        Args:
            frame: RGB frame from webcam
            alpha: blending factor (0=black, 1=frame)
        Returns:
            Blended frame
        """
        black = np.zeros_like(frame)
        # Use Gaussian blur for soft edges
        blurred = cv2.GaussianBlur(frame, (35, 35), 0)
        blended = cv2.addWeighted(black, 1-alpha, blurred, alpha, 0)
        return blended

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
