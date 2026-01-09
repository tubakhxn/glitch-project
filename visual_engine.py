"""
visual_engine.py
Generates and renders neon HUD-style shapes with smoothing, perspective, and parallax.
"""
import numpy as np
import cv2
import math

NEON_COLOR = (0, 255, 255)  # Neon yellow/amber in RGB

class VisualEngine:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.shape_state = {
            "scale": 1.0,
            "angle": 0.0,
            "depth": 0.5,
        }
        self.smoothing = 0.15

    def update(self, gesture_left, gesture_right):
        # Smoothly update shape state
        self.shape_state["scale"] += self.smoothing * (gesture_left.get("scale", 1.0) - self.shape_state["scale"])
        self.shape_state["angle"] += self.smoothing * (gesture_right.get("angle", 0.0) - self.shape_state["angle"])
        self.shape_state["depth"] += self.smoothing * (gesture_right.get("depth", 0.5) - self.shape_state["depth"])

    def render(self, frame):
        # Draw futuristic HUD shapes with perspective and parallax
        center = (int(self.width//2), int(self.height//2))
        scale = self.shape_state["scale"]
        angle = self.shape_state["angle"]
        depth = self.shape_state["depth"]
        # Perspective math for depth cues
        z = 200 + 400 * depth
        radius = int(120 * scale * (1 + 0.5 * depth))
        # Neon circle
        cv2.circle(frame, center, radius, NEON_COLOR, 4, lineType=cv2.LINE_AA)
        # Neon ring (arc)
        start_angle = int(np.degrees(angle))
        end_angle = start_angle + 270
        cv2.ellipse(frame, center, (radius+30, radius+30), 0, start_angle, end_angle, NEON_COLOR, 2, lineType=cv2.LINE_AA)
        # Radial HUD elements
        for i in range(8):
            theta = angle + i * math.pi/4
            x = int(center[0] + (radius+60) * math.cos(theta))
            y = int(center[1] + (radius+60) * math.sin(theta) * (1 + 0.3 * depth))
            cv2.line(frame, center, (x, y), NEON_COLOR, 2, lineType=cv2.LINE_AA)
        # Glow effect (temporal smoothing)
        frame = cv2.GaussianBlur(frame, (0, 0), sigmaX=8, sigmaY=8)
        return frame
