import sys
from Camera import Camera
import cv2
import os

from flask import Response, Flask, render_template

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def indexPage():
    return render_template('index.html')


def gen(camera):
    while True:
        try:
            frame = camera.getFrame()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        except Exception as e:
            print(e)


@app.route('/video_feed')
def video_feed():
    return Response(gen(Camera(source=0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
