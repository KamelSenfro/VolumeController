import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import signal
import sys

# Initialize MediaPipe and Pycaw
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Audio control setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]
volBar, volPer = 400, 0

# Webcam setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Signal handler for clean exit
def signal_handler(sig, frame):
    print('Terminating...')
    cam.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Main function
def main():
    """Main loop for hand detection and volume control."""

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75
    ) as hands:

        prev_time = 0

        while cam.isOpened():
            success, image = cam.read()
            if not success:
                continue

            # Convert BGR image to RGB for MediaPipe processing
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            # Initialize list for landmarks
            lmList = []

            if results.multi_hand_landmarks:
                # Assuming only one hand is detected
                myHand = results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHand.landmark):
                    h, w, _ = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Volume control logic based on thumb and index finger distance
                if lmList:
                    x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                    x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                    # Draw landmarks for visualization (optional)
                    cv2.circle(image, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
                    cv2.circle(image, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green line between fingertips

                    length = math.hypot(x2 - x1, y2 - y1)  # Calculate distance between fingertips

                    # Mute audio when distance is very small (fingers close together)
                    if length < 30:
                        volume.SetMasterVolumeLevel(minVol, None)
                        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line for mute indication

                    # Adjust volume proportionally based on fingertip distance (larger distance = higher volume)
                    else:
                        vol = np.interp(length, [30, 220], [minVol, maxVol])  # Linear interpolation for volume range
                        volume.SetMasterVolumeLevel(vol, None)
                        volBar = np.interp(length, [30, 220], [400, 150])  # Visual volume bar position
                        volPer = np.
