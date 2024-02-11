import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
import numpy as np
from email.mime.image import MIMEImage
import time
from flask import Flask, render_template, request, Response
app = Flask(__name__)

donel = False
doner = False
x1, y1, x2, y2 = 0, 0, 0, 0
movement_threshold = 5  # Number of movements required to trigger an email
time_threshold = 10  # Time threshold in seconds for continuous motion
movement_counter = 0
last_notification_time = time.time()

# Logging
log_file = open("motion_detection_log.txt", "a")

def send_email(image_data):
    global last_notification_time
    msg = MIMEMultipart()
    msg['Subject'] = 'Motion Detected'
    msg['From'] = 'theftdetector84@gmail.com'
    msg['To'] = 'theftmailcheck@gmail.com'

    image = MIMEImage(image_data, name='motion_detection.jpg')
    msg.attach(image)

    smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
    smtp_server.starttls()
    smtp_server.login('theftdetector84@gmail.com', 'yhfepxqoyswoiatn')  # Replace 'your_password' with the actual password
    smtp_server.sendmail('theftdetector84@gmail.com', 'theftmailcheck@gmail.com', msg.as_string())
    smtp_server.quit()

def select(event, x, y, flags, param):
    global x1, y1, x2, y2, donel, doner
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        donel = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        x2, y2 = x, y
        doner = True
        print(doner, donel)

def detect_objects(frame):
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    with open("yolo/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.resize(frame, None, fx=2.0, fy=2.0)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    confidences = []
    class_ids = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1 / 2, color, 2)

    return frame


def rect_noise(use_time_threshold=True):
    global x1, y1, x2, y2, donel, doner, movement_counter, last_notification_time
    cap = cv2.VideoCapture(0)

    # Set a lower resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("select_region")
    cv2.setMouseCallback("select_region", select)

    # Create a background subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2()

    while True:
        _, frame = cap.read()
        cv2.imshow("select_region", frame)

        if cv2.waitKey(1) == 27 or doner:
            cv2.destroyAllWindows()
            break

    while True:
        _, frame1 = cap.read()
        _, frame2 = cap.read()

        frame1only = frame1[y1:y2, x1:x2]
        frame2only = frame2[y1:y2, x1:x2]

        diff = cv2.absdiff(frame2only, frame1only)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame1, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                cv2.putText(frame1, "Motion Detected", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                motion_detected = True
                break

        cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #frame1 = detect_objects(frame1)
        cv2.imshow("esc. to exit", frame1)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

        if motion_detected:
            movement_counter += 1
            current_time = time.time()

            if use_time_threshold:
                if current_time - last_notification_time >= time_threshold:
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    send_email(frame_bytes)
                    last_notification_time = current_time  # Update the notification time

                    log_file.write(f"Motion detected at {time.ctime()} - Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")
                    

            else:
                if movement_counter >= movement_threshold:
                    _, buffer = cv2.imencode('.jpg', frame1)
                    frame_bytes = buffer.tobytes()
                    send_email(frame_bytes)
                    movement_counter = 0  # Reset the counter after sending the email

                    log_file.write(f"Motion detected at {time.ctime()} - Threshold: {movement_threshold} "
                                   f"Time Threshold: {time_threshold}\n")
                    

    log_file.close()  # Close the log file when finished
rect_noise(use_time_threshold=False)
    


