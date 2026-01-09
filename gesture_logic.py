"""
gesture_logic.py
Interprets hand keypoints into gestures and control signals for both hands.
"""
import numpy as np
import math

class GestureLogic:
    def __init__(self):
        self.smoothing = 0.2
        self.last_bbox = [300, 200, 200, 200]  # x, y, w, h
        self.last_offset = [0, 0]

    def update(self, hands):
        # Left hand: index/thumb define bbox (top-left, bottom-right)
        # Right hand: index/thumb midpoint defines offset
        bbox = self.last_bbox.copy()
        offset = self.last_offset.copy()
        for hand in hands:
            keypoints = hand["keypoints"]
            if hand["hand"] == "left" and len(keypoints) >= 2:
                pt1 = np.array(keypoints[0][:2])
                pt2 = np.array(keypoints[1][:2])
                x0, y0 = pt1
                x1, y1 = pt2
                x, y = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)
                bbox[0] += self.smoothing * (x - bbox[0])
                bbox[1] += self.smoothing * (y - bbox[1])
                bbox[2] += self.smoothing * (w - bbox[2])
                bbox[3] += self.smoothing * (h - bbox[3])
            elif hand["hand"] == "right" and len(keypoints) >= 2:
                pt1 = np.array(keypoints[0][:2])
                pt2 = np.array(keypoints[1][:2])
                mid = (pt1 + pt2) / 2
                offset[0] += self.smoothing * (mid[0] - offset[0])
                offset[1] += self.smoothing * (mid[1] - offset[1])
        self.last_bbox = bbox
        self.last_offset = offset
        return {"bbox": bbox, "offset": offset}
