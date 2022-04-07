import cv2
import time

from cv2 import RETR_EXTERNAL

first_frame = None  # creates the first frame condition with just the background

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    thresh_delta = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta, None, iterations=2)

    (cnts, _) = cv2.findContours(thresh_delta.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contours in cnts:
        if cv2.contourArea(contours) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contours)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow('Gray frame', gray)
    cv2.imshow('Delta framing', delta_frame)
    cv2.imshow('Threshold frame', thresh_delta)
    cv2.imshow('Color frame', frame)

    key = cv2.waitKey(1)
    # print(gray)
    # print(delta_frame)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows
