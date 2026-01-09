"""
main_glitch.py
Pipeline for real-time face glitch effect controlled by hand gestures.
"""
import cv2
import numpy as np
from video_capture import VideoCapture
from hand_tracking import HandTracker
from gesture_logic import GestureLogic
from glitch_effect import apply_glitch

FPS = 30

def main():
    cap = VideoCapture()
    hand_tracker = HandTracker()
    gesture_logic = GestureLogic()
    # Smoothing state
    smooth_bbox = [0, 0, 0, 0]
    smooth_offset = [0, 0]
    alpha = 0.25
    while True:
        frame = cap.get_frame()
        hands = hand_tracker.detect_hands(frame)
        gestures = gesture_logic.update(hands)
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        left = right = None
        for hand in hands:
            if hand["hand"] == "left":
                left = hand
            elif hand["hand"] == "right":
                right = hand
        if left:
            lkps = left["keypoints"]
            lx0, ly0 = int(lkps[0][0]), int(lkps[0][1])
            lx1, ly1 = int(lkps[1][0]), int(lkps[1][1])
            # Rectangle from left hand
            x_min, x_max = min(lx0, lx1), max(lx0, lx1)
            y_min, y_max = min(ly0, ly1), max(ly0, ly1)
            # Default offset
            dx, dy = 0, 0
            if right:
                rkps = right["keypoints"]
                rx0, ry0 = int(rkps[0][0]), int(rkps[0][1])
                rx1, ry1 = int(rkps[1][0]), int(rkps[1][1])
                # Use midpoint between right index and thumb as offset
                dx = int(((rx0 + rx1) // 2) - w // 2)
                dy = int(((ry0 + ry1) // 2) - h // 2)
            # Temporal smoothing
            smooth_bbox[0] = int((1-alpha)*smooth_bbox[0] + alpha*x_min)
            smooth_bbox[1] = int((1-alpha)*smooth_bbox[1] + alpha*y_min)
            smooth_bbox[2] = int((1-alpha)*smooth_bbox[2] + alpha*(x_max-x_min))
            smooth_bbox[3] = int((1-alpha)*smooth_bbox[3] + alpha*(y_max-y_min))
            smooth_offset[0] = int((1-alpha)*smooth_offset[0] + alpha*dx)
            smooth_offset[1] = int((1-alpha)*smooth_offset[1] + alpha*dy)
            # Apply smoothed values
            sx, sy, sw, sh = smooth_bbox
            sdx, sdy = smooth_offset
            x_min = max(0, sx + sdx)
            x_max = min(w, sx + sw + sdx)
            y_min = max(0, sy + sdy)
            y_max = min(h, sy + sh + sdy)
            # Only draw if box is at least 2x2
            if x_max > x_min+1 and y_max > y_min+1:
                mask[y_min:y_max, x_min:x_max] = 1
            box_w = max(1, x_max - x_min)
            amount = np.clip(box_w / w, 0.05, 1.0)  # allow very small squares
            shift = max(2, int(box_w * 0.2))
        else:
            amount = 0.7
            shift = 30
        noise = 0.3
        glitched = apply_glitch(frame, mask, amount=amount, shift=shift, noise=noise)
        out = cv2.addWeighted(frame, 0.5, glitched, 0.5, 0)
        cv2.imshow("Glitch Face Art", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(int(1000/FPS)) & 0xFF == 27:
            break
    cap.release()

if __name__ == "__main__":
    main()