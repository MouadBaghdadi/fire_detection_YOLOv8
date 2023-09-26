from ultralytics import YOLO
import cvzone
import cv2
import math

cap = cv2.VideoCapture('fire.mp4')
model = YOLO('best.pt')

classnames = ['Fire']

while True:
    _ , frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    result = model(frame,stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            conf = box.conf[0]
            conf = math.ceil(conf*100)

            if conf>20:
                x1,y1,x2,y2 =box.xyxy[0]
                x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
                # names = box.cls[0]
                # names = int(names)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 5)
                cvzone.putTextRect(frame, f'{classnames[0]}{conf}%', [x1+8,y1+100], scale = 1.5, thickness = 2)

    cv2.imshow('frame2', frame)
    cv2.waitKey(1)