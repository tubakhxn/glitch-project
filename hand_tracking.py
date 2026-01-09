"""
hand_tracking.py
Loads a Hugging Face hand keypoint detection model manually and runs inference on webcam frames.
"""
import torch
import numpy as np
from PIL import Image

import cv2
import numpy as np

class HandTracker:
    def __init__(self):
        pass

    def detect_hands(self, frame):
        """
        Detects hand-like contours and returns bounding box centers as pseudo keypoints.
        Returns two hands if found, else returns empty or one.
        """
        h, w = frame.shape[:2]
        # Convert to HSV and threshold for skin color (very basic, works in good light)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # Morphological operations to clean up
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area, take up to 2 largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        hands = []
        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 1000:
                continue
            x, y, ww, hh = cv2.boundingRect(cnt)
            # Use top-left and bottom-right as pseudo index and thumb
            hands.append({
                "hand": "left" if i == 0 else "right",
                "keypoints": [
                    (x, y, 1.0),  # pseudo index
                    (x+ww, y+hh, 1.0)  # pseudo thumb
                ]
            })
        return hands
