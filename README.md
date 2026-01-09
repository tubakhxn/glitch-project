# Glitch Face Art (by tubakhxn)

A real-time interactive computer vision art project using OpenCV. Control a glitch effect on your webcam feed with hand gestures—no deep learning required!

## Features
- **Hand gesture control:**
  - Left hand (index + thumb): Pinch/spread to set the size and position of the glitch mask.
  - Right hand (index + thumb): Move to smoothly translate the glitch mask across the screen.
- **Glitch effect:**
  - Datamosh-style distortion applied only inside the mask.
  - Mask movement and resizing are smoothed for a professional, non-jittery look.
- **No deep learning required:**
  - Uses OpenCV hand contour detection for gesture input.
- **Modular codebase:**
  - Easy to extend or remix for your own creative effects.

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```bash
   python main_glitch.py
   ```

## Fork & Remix
- Fork this repo to create your own interactive webcam art.
- Add new effects, change the gesture logic, or integrate with other creative tools.
- Credit: Created by tubakhxn

---
**For experimental art, live visuals, and creative webcam effects.**
