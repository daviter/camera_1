import os
import cv2
import time
from datetime import datetime
from flask import Flask, render_template, Response, redirect, url_for, request, send_from_directory
from deepface import DeepFace
import pygame

app = Flask(__name__)
camera = None
camera_on = False

DB_PATH = os.path.join(os.getcwd(), 'known_faces')
UNKNOWN_PATH = os.path.join(os.getcwd(), 'unknown_faces')
os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(UNKNOWN_PATH, exist_ok=True)


# âm thanh cảnh báo
def play_alert_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("tieng_coi_hu_dj-www_tiengdong_com.mp3")  # đổi tên file nếu muốn
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)


# thử tải MTCNNWrapper (DeepFace >= 2024)
try:
    from deepface.detectors import MTCNNWrapper

    has_wrapper = True
except ImportError:
    has_wrapper = False


def gen_frames():
    global camera
    detector = None
    if has_wrapper:
        print("✅ Dùng MTCNNWrapper (DeepFace >=2024)")
        detector = MTCNNWrapper.build_model()
    else:
        print("⚠️ Không có MTCNNWrapper. Sẽ để DeepFace tự xử lý detector.")

    unknown_active = False
    last_unknown_time = 0
    UNKNOWN_DELAY = 30  # giây

    while camera_on:
        success, frame = camera.read()
        if not success:
            break

        detected_unknown = False
        now = time.time()

        try:
            if has_wrapper:
                # bản mới: tự gọi MTCNNWrapper
                face_objs = MTCNNWrapper.detect_faces(detector, frame, align=False)
            else:
                # bản cũ: không có wrapper, cứ để DeepFace detect
                face_objs = []
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_img = frame[y:y + h, x:x + w]
                    face_objs.append((face_img, (x, y, w, h)))

            for face_img, (x, y, w, h) in face_objs:
                tmp_path = "temp_face.jpg"
                cv2.imwrite(tmp_path, face_img)

                name = "Unknown"

                result = DeepFace.find(
                    img_path=tmp_path,
                    db_path=DB_PATH,
                    enforce_detection=False,
                    detector_backend="mtcnn"  # để DeepFace tự xử lý
                )

                if result and len(result[0]) > 0:
                    df = result[0]
                    identity_path = df.iloc[0]['identity']
                    distance = df.iloc[0]['distance']

                    threshold = 0.3
                    if distance <= threshold:
                        name = os.path.basename(identity_path)
                    else:
                        detected_unknown = True
                else:
                    detected_unknown = True

                os.remove(tmp_path)

                color = (0, 0, 255) if detected_unknown else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if detected_unknown:
                if not unknown_active or (now - last_unknown_time >= UNKNOWN_DELAY):
                    unknown_active = True
                    last_unknown_time = now

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(UNKNOWN_PATH, f"unknown_{ts}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"📸 Lưu ảnh người lạ: {save_path}")
                    play_alert_sound()
            else:
                if unknown_active:
                    print("✅ Không còn người lạ. Quay lại bình thường.")
                    unknown_active = False

        except Exception as e:
            print("❌ Lỗi phát hiện:", e)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', camera_on=camera_on)


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_camera')
def toggle_camera():
    global camera, camera_on
    if camera_on:
        camera_on = False
        if camera:
            camera.release()
            camera = None
    else:
        camera = cv2.VideoCapture(1)
        camera_on = True
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    filepath = os.path.join(DB_PATH, file.filename)
    file.save(filepath)
    return redirect(url_for('index'))


@app.route('/unknown_faces')
def unknown_faces():
    files = os.listdir(UNKNOWN_PATH)
    files = sorted(files, reverse=True)
    return render_template("unknown_faces.html", files=files)


@app.route('/unknown_faces/<filename>')
def unknown_file(filename):
    return send_from_directory(UNKNOWN_PATH, filename)


@app.route('/shutdown', methods=['POST'])
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        os._exit(0)
    func()
    return 'Đã tắt server!'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
