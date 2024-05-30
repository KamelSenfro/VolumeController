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

# Signal handler to ensure clean exit
def signal_handler(sig, frame):
    print('Terminating...')
    cam.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)

# Optimized Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75) as hands:

    prev_time = 0

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            continue

        # Convert the BGR image to RGB and process with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Initialize the list for landmarks
        lmList = []

        if results.multi_hand_landmarks:
            # Assuming only one hand is detected
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, _ = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Volume control logic
            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                # Draw markers only when landmarks are detected
                cv2.circle(image, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
                cv2.circle(image, (x2, y2), 10, (255, 255, 255), cv2.FILLED)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                length = math.hypot(x2 - x1, y2 - y1)
                if length < 50:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                vol = np.interp(length, [50, 220], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                volBar = np.interp(length, [50, 220], [400, 150])
                volPer = np.interp(length, [50, 220], [0, 100])

                # Draw volume bar and percentage
                bar_x = wCam - 85
                cv2.rectangle(image, (bar_x, 150), (bar_x + 35, 400), (255, 255, 255), 3)
                cv2.rectangle(image, (bar_x, int(volBar)), (bar_x + 35, 400), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, f'{int(volPer)} %', (bar_x - 10, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 3)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(image, f'FPS: {int(fps)}', (wCam - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the image
        cv2.imshow('Hand Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
