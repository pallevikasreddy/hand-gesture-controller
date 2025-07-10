import cv2
import mediapipe as mp
import numpy as np
from screen_brightness_control import set_brightness
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Volume Control Setup
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_min, vol_max = volume.GetVolumeRange()[:2]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Use tip of thumb and index finger
            x1, y1 = lm_list[4]   # Thumb tip
            x2, y2 = lm_list[8]   # Index tip
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))

            # Map length (30 to 200) to volume range
            vol = np.interp(length, [30, 200], [vol_min, vol_max])
            volume.SetMasterVolumeLevel(vol, None)

            # Brightness based on Y position of hand center
            hand_y = np.mean([pt[1] for pt in lm_list])
            brightness = np.interp(hand_y, [0, img.shape[0]], [100, 0])
            set_brightness(int(brightness))

            cv2.putText(img, f'Vol: {int(np.interp(length, [30, 200], [0, 100]))}%', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(img, f'Brightness: {int(brightness)}%', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    cv2.imshow("Hand Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
