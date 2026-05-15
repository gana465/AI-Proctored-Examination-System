from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import threading
import os
import time

app = Flask(__name__)

# =========================================
# FACE DETECTOR
# =========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# =========================================
# CAMERA
# =========================================

camera = cv2.VideoCapture(0)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =========================================
# GLOBAL VARIABLES
# =========================================

alerts = 0
status = "Normal"

face_status = "Detected"
motion_status = "No"
block_status = "No"

FRAME_LIMIT = 20
MOTION_AREA = 6000
BRIGHTNESS_THRESHOLD = 35
MAX_ALERTS = 3

no_face_frames = 0
multi_face_frames = 0
away_frames = 0
motion_frames = 0
blocked_frames = 0

prev_gray = None

# =========================================
# VIDEO STREAM
# =========================================

def generate_frames():

    global alerts
    global status

    global face_status
    global motion_status
    global block_status

    global no_face_frames
    global multi_face_frames
    global away_frames
    global motion_frames
    global blocked_frames

    global prev_gray

    while True:

        success, frame = camera.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =========================================
        # CAMERA BLOCK DETECTION
        # =========================================

        brightness = np.mean(gray)

        if brightness < BRIGHTNESS_THRESHOLD:

            blocked_frames += 1

            block_status = "Yes"

            if blocked_frames >= FRAME_LIMIT:

                alerts += 1

                blocked_frames = 0

                status = "Camera Blocked"

        else:

            blocked_frames = 0

            block_status = "No"

        # =========================================
        # FACE DETECTION
        # =========================================

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) == 0:

            face_status = "No Face"

            no_face_frames += 1

            if no_face_frames >= FRAME_LIMIT:

                alerts += 1

                no_face_frames = 0

                status = "No Face"

        elif len(faces) > 1:

            face_status = "Multiple Faces"

            multi_face_frames += 1

            if multi_face_frames == FRAME_LIMIT:

                alerts += 1

                status = "Multiple Faces"

        else:

            no_face_frames = 0
            multi_face_frames = 0

            (x, y, w, h) = faces[0]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            center_x = x + w // 2

            if center_x < 200 or center_x > 440:

                away_frames += 1

                face_status = "Looking Away"

                if away_frames >= FRAME_LIMIT:

                    alerts += 1

                    away_frames = 0

                    status = "Looking Away"

            else:

                away_frames = 0

                face_status = "Detected"

        # =========================================
        # MOTION DETECTION
        # =========================================

        blur = cv2.GaussianBlur(gray, (15, 15), 0)

        if prev_gray is not None:

            diff = cv2.absdiff(prev_gray, blur)

            thresh = cv2.threshold(
                diff,
                25,
                255,
                cv2.THRESH_BINARY
            )[1]

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            motion_detected = False

            for contour in contours:

                if cv2.contourArea(contour) > MOTION_AREA:

                    motion_detected = True

                    x, y, w, h = cv2.boundingRect(contour)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x+w, y+h),
                        (0, 255, 0),
                        2
                    )

            if motion_detected:

                motion_frames += 1

                motion_status = "Yes"

                if motion_frames >= FRAME_LIMIT:

                    alerts += 1

                    motion_frames = 0

                    status = "Excessive Motion"

            else:

                motion_status = "No"

                motion_frames = 0

        prev_gray = blur

        # =========================================
        # TERMINATE EXAM
        # =========================================

        if alerts >= MAX_ALERTS:

            status = "EXAM TERMINATED"

            cv2.putText(
                frame,
                "EXAM TERMINATED",
                (100, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                4
            )

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame +
                b'\r\n'
            )

            camera.release()
            cv2.destroyAllWindows()

            break

        # =========================================
        # DISPLAY TEXT
        # =========================================

        cv2.putText(
            frame,
            f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Alerts: {alerts}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =========================================
# HOME PAGE
# =========================================

@app.route('/')
def home():
    return render_template('index.html')

# =========================================
# VIDEO FEED
# =========================================

@app.route('/video_feed')
def video_feed():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# =========================================
# STATUS API
# =========================================

@app.route('/status')
def get_status():

    return jsonify({

        'status': status,
        'alerts': alerts,
        'face': face_status,
        'motion': motion_status,
        'blocked': block_status
    })

# =========================================
# RUN APP
# =========================================

if __name__ == '__main__':

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        threaded=True
    )
