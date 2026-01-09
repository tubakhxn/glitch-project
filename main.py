"""
main.py
Orchestrates the real-time loop, blending webcam feed, processing gestures, and rendering visuals.
"""
import cv2
import numpy as np
from video_capture import VideoCapture
from hand_tracking import HandTracker
from gesture_logic import GestureLogic
from visual_engine import VisualEngine

FPS = 60

if __name__ == "__main__":
    cap = VideoCapture()
    tracker = HandTracker()
    gesture = GestureLogic()
    engine = VisualEngine()

    while True:
        frame = cap.get_frame()
        blended = cap.blend_with_black(frame, alpha=0.25)
        hands = tracker.detect_hands(frame)
        gestures = gesture.update(hands)
        engine.update(gestures["left"], gestures["right"])
        output = engine.render(blended.copy())
        # Convert RGB to BGR for OpenCV display
        cv2.imshow("Neon HUD Vision", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(int(1000/FPS)) & 0xFF == 27:
            break
    cap.release()
