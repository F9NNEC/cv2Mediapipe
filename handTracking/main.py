import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands # module
hands = mpHands.Hands()
mpdraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # BGR(openCV) to RBG(mediapipe)
    results = hands.process(imgRGB) # detect hands
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks: # chek hands
        for handLms in results.multi_hand_landmarks: # for each detected hands
            mpdraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # draw 21 landmarks and connections line

    mirroredImg = cv2.flip(img,1) # 0 vertical, 1 horizontal
    cv2.imshow('img',mirroredImg)
    cv2.waitKey(1)