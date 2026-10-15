from flask import Flask, render_template, request, url_for, session, redirect, flash, Response
import numpy as np
import os
import cv2
import re

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = 'static/tests/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = model_from_json(open("models.json", "r").read())
model.load_weights('model_weight.h5')

def allowed_file(filename):
    return ('.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)

def gen_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels = np.repeat(img_pixels, 3, axis=-1)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions, axis=1)

                drowsy = ('Drowsiness', 'Normal', 'Normal', 'Drowsiness')
                predicted_drowsy = drowsy[max_index[0]]
                cv2.putText(frame, predicted_drowsy, (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

app.secret_key = "your_secret_key"

# ---------------------------------------------------
#                LOGIN WITHOUT DATABASE
# ---------------------------------------------------
@app.route('/first')
def first():
    return render_template('first.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "admin":
            session['loggedin'] = True
            session['username'] = "admin"
            return redirect(url_for('index'))
        else:
            msg = "Incorrect username or password!"

    return render_template("login.html", msg=msg)

# ---------------------------------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/index")
def index():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template("index.html")

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(48, 48))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0

    predict_x = model.predict(test_image)
    preds = np.argmax(predict_x, axis=1)

    drowsy = ('Drowsiness', 'Normal', 'Normal', 'Drowsiness')
    result = drowsy[preds[0]]
    return result, predict_x

@app.route('/uploadimage', methods=['POST'])
def uploadimage():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_path)

        predict_result, predict_x = predict_label(img_path)

        return render_template("index.html",
                               prediction=predict_result,
                               img_path=img_path,
                               predict_x=predict_x[0][0])

    return render_template('index.html')

@app.route("/performance")
def performance():
    if 'loggedin' not in session:
        return redirect(url_for('login'))
    return render_template("performance.html")

@app.route('/chart3')
def chart3():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    # No database → use sample dummy values
    labels = ["Drowsiness", "Normal"]
    values = [10, 20]
    legend = "Review Summary"

    return render_template('chart3.html', values=values, labels=labels, legend=legend)

if __name__ == '__main__':
    app.run()
