"""
face_segmentation.py
Loads a Hugging Face face segmentation model and returns a binary mask for the face region.
"""
import cv2
import numpy as np

class FaceSegmenter:
    def __init__(self):
        # Use OpenCV's built-in Haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def get_face_mask(self, frame):
        """
        Returns a binary mask (np.uint8) of the face region using Haar cascade.
        Args:
            frame: RGB numpy array
        Returns:
            mask: np.ndarray (0=background, 1=face)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in faces:
            cv2.ellipse(mask, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, 1, -1)
        return mask
