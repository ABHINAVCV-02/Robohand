import cv2
import mediapipe as mp
import serial
import time

# ---------- SERIAL ----------
arduino = serial.Serial('COM3', 9600)  # CHANGE COM PORT
time.sleep(2)

# ---------- MEDIAPIPE ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

last_command = None
COOLDOWN = 1.5   # seconds (prevents jitter)
last_time = 0

def count_fingers(hand_landmarks):
    fingers = 0

    # Index finger
    if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
        fingers += 1
    # Middle finger
    if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
        fingers += 1
    # Ring finger
    if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
        fingers += 1
    # Pinky finger
    if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
        fingers += 1

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)

            current_time = time.time()

            # ✌ TWO FINGERS → MOVE TO 90°
            if finger_count == 2 and last_command != "UP":
                if current_time - last_time > COOLDOWN:
                    print("✌ Two fingers detected → Servo to 90°")
                    arduino.write(b'U')
                    last_command = "UP"
                    last_time = current_time

            # ☝ ONE FINGER → RETURN TO 0°
            elif finger_count == 1 and last_command != "DOWN":
                if current_time - last_time > COOLDOWN:
                    print("☝ One finger detected → Servo to 0°")
                    arduino.write(b'D')
                    last_command = "DOWN"
                    last_time = current_time

            cv2.putText(frame, f"Fingers: {finger_count}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.putText(frame, "1 finger = 0°, 2 fingers = 90°",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)

    cv2.imshow("Finger Gesture Servo Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
