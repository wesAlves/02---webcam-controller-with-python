from datetime import datetime
import cv2
import time
import pandas


first_frame = None  # creates the first frame condition with just the background

status_list = [None, None]
times = []

data_frames = pandas.DataFrame(columns=['Start', 'End'])

video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    status = 0

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
        if cv2.contourArea(contours) < 100:
            continue

        status = 1

        (x, y, w, h) = cv2.boundingRect(contours)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow('Gray frame', gray)
    cv2.imshow('Delta framing', delta_frame)
    cv2.imshow('Threshold frame', thresh_delta)
    cv2.imshow('Color frame', frame)

    key = cv2.waitKey(1)
    # print(gray)
    # print(delta_frame)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# print(status_list)
# print(times)

for time in range(0, len(times), 2):
    data_frames = data_frames.append(
        {'Start': times[time], 'End': times[time+1]}, ignore_index=True)

data_frames.to_csv('Times.csv')


video.release()
cv2.destroyAllWindows
