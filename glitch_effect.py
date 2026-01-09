"""
glitch_effect.py
Applies a glitch effect to a masked region of an image, controlled by parameters.
"""
import numpy as np
import cv2

def apply_glitch(frame, mask, amount=0.5, shift=20, noise=0.3):
    """
    Applies a datamosh/glitch effect to the masked region.
    Args:
        frame: RGB numpy array
        mask: np.ndarray (0=background, 1=face)
        amount: float, how much to distort
        shift: int, max pixel shift for blocks
        noise: float, how much color noise to add
    Returns:
        Glitched frame (RGB)
    """
    out = frame.copy()
    # Find bounding box of mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return out
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    face = out[y0:y1, x0:x1].copy()
    h, w = face.shape[:2]
    # Block shift
    for i in range(0, h, 10):
        if np.random.rand() < amount:
            dx = int((np.random.rand() - 0.5) * 2 * shift)
            face[i:i+10] = np.roll(face[i:i+10], dx, axis=1)
    # Color channel shift
    if amount > 0.2:
        for c in range(3):
            face[..., c] = np.roll(face[..., c], int((np.random.rand()-0.5)*shift), axis=0)
    # Add noise
    if noise > 0:
        noise_map = (np.random.randn(*face.shape) * 32 * noise).astype(np.int16)
        face = np.clip(face.astype(np.int16) + noise_map, 0, 255).astype(np.uint8)
    # Composite back
    out[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0] = face[mask[y0:y1, x0:x1] > 0]
    return out
