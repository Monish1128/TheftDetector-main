from flask import Flask, render_template, request, Response
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import time

app = Flask(__name__)

donel = False
doner = False
x1, y1, x2, y2 = 0, 0, 0, 0
movement_counter = 0
last_notification_time = time.time()

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

def generate_frames(motion_threshold, use_time_threshold, as_motion_threshold, time_threshold):
    global x1, y1, x2, y2, donel, doner, movement_counter, last_notification_time
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("select_region")
    cv2.setMouseCallback("select_region", select)

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
        ret, buffer = cv2.imencode('.jpg', frame1)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

        if motion_detected:
            movement_counter += 1
            current_time = time.time()

            if current_time - last_notification_time >= time_threshold:
                _, buffer = cv2.imencode('.jpg', frame1)
                frame_bytes = buffer.tobytes()
                send_email(frame_bytes)
                last_notification_time = current_time
    log_file.close()

@app.route('/')
def index():
    return render_template('motion_detection.html')

@app.route('/start_detection_feed', methods=['POST'])
def start_detection():
    as_motion_threshold = int(request.form['as_motion_threshold'])
    time_threshold = int(request.form['time_threshold'])

    motion_threshold = request.form.get('use_as_threshold') == 'true'
    use_time_threshold = request.form.get('use_time_threshold') == 'true'

    return render_template('video_feed.html', motion_threshold=motion_threshold, use_time_threshold=use_time_threshold, as_motion_threshold=as_motion_threshold, time_threshold=time_threshold)

@app.route('/video_feed')
def video_feed():
    motion_threshold = request.args.get('motion_threshold') == 'true'
    use_time_threshold = request.args.get('use_time_threshold') == 'true'
    as_motion_threshold = int(request.args.get('as_motion_threshold'))
    time_threshold = int(request.args.get('time_threshold'))

    return Response(generate_frames(motion_threshold, use_time_threshold, as_motion_threshold, time_threshold), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
