import cv2
import numpy as np
import mediapipe as mp
import random
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

snake = [(300,300)]
snake_length = 20
snake_direction = (0,0)
food = (random.randint(100,500),random.randint(100,400))
score = 0

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_RGB)

    # update snake direction based on hand landmark
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lms = hand_landmarks.landmark
            index_tip = lms[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            snake_direction = (x - snake[0][0], y - snake[0][1])
            length = math.hypot(*snake_direction)
            if length != 0:
                snake_direction = (snake_direction[0] / length * 10, snake_direction[1] / length * 10)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # move snake
    if snake_direction != (0, 0):
        new_head = (int(snake[0][0] + snake_direction[0]), int(snake[0][1] + snake_direction[1]))
        snake.insert(0, new_head)
        if len(snake) > snake_length:
            snake.pop()

    # check for food collision
    if math.hypot(snake[0][0] - food[0], snake[0][1] - food[1]) < 20:
        food = (random.randint(50, 590), random.randint(50, 430))
        score += 1
        snake_length += 5

    # drawing
    cv2.circle(img, food, 10, (0, 0, 255), -1)
    for point in snake:
        cv2.circle(img, point, 10, (0, 255, 0), -1)
    cv2.putText(img, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # check boundaries
    if not (0 < snake[0][0] < 640 and 0 < snake[0][1] < 480):
        cv2.putText(img, "Game Over", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        cv2.imshow("Snake Game", img)
        cv2.waitKey(3000)
        break

    cv2.imshow("Snake Game", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()