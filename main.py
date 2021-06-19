import cv2
import mediapipe as mp
import time


# Hand Tracking
def hand_tracking():
    # Load models
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize input
    cap = cv2.VideoCapture(0)

    # Initialize FPS
    prev_time = 0

    while True:
        success, img = cap.read()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        # print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, landmark in enumerate(hand_landmarks.landmark):
                    # print(i, landmark)
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    height, width, channels = img.shape
                    px, py = int(landmark.x * width), int(landmark.y * height)
                    if i == 0:
                        cv2.circle(img, (px, py), 15, (255, 0, 255))

        curr_time = time.time()
        delta_time = curr_time - prev_time
        fps = 1 // delta_time
        prev_time = curr_time
        cv2.putText(img, str(fps), (12, 65), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (128,128,192), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


def main():
    hand_tracking()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
