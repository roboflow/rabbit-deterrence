import json
import cv2
import base64
import requests
import io
from PIL import Image
import pygame as pg
import os

# load config
with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    LOCAL_SERVER = config["LOCAL_SERVER"]
    DATASET_NAME = config["DATASET_NAME"]

    FRAMERATE = config["FRAMERATE"]
    BUFFER = config["BUFFER"]

# Local Server Link
infer_url = "".join([
    "https://detect.roboflow.com/" + ROBOFLOW_MODEL,
    "?api_key=" + ROBOFLOW_API_KEY,
    "&name=YOUR_IMAGE.jpg"
])
if not LOCAL_SERVER:
    pass
else:
    infer_url = "".join([
        "http://127.0.0.1:9001/" + ROBOFLOW_MODEL,
        "?api_key=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)


# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = ROBOFLOW_SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get predictions from Roboflow Infer API
    resp = requests.post(infer_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).json()['predictions']

    # # Draw all predictions
    # if len(resp) > 0:
    #     success = False
    #     # Prevent Blurry Images from being uploaded
    #     if cv2.Laplacian(img, cv2.CV_64F).var() > 40:
    #         success, imageId = uploadImage(img)
    #     playSound(os.getcwd() + os.sep + 'crying.mp3')
    #
    #     if success:
    #         uploadAnnotation(imageId, resp)

    for prediction in resp:
        writeOnStream(prediction['x'], prediction['y'], prediction['width'], prediction['height'],
                      prediction['class'],
                      img)

    return img


def writeOnStream(x, y, width, height, className, frame):
    # Draw a Rectangle around detected image
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                  (255, 0, 0), 2)

    # Draw filled box for class name
    cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                  (255, 0, 0), cv2.FILLED)

    # Set label font + draw Text
    font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)


def playSound(soundFile):
    '''
    stream music with mixer.music module in blocking manner
    this will stream the sound from disk while playing
    '''
    clock = pg.time.Clock()
    pg.mixer.init()
    pg.mixer.music.load(soundFile)
    print("Music file {} loaded!".format(soundFile))

    pg.mixer.music.play()

    while pg.mixer.music.get_busy():
        clock.tick(30)


def uploadImage(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImage = Image.fromarray(frame)
    # Convert to JPEG Buffer
    buffered = io.BytesIO()
    pilImage.save(buffered, quality=90, format="JPEG")

    # Base 64 Encode
    img_str = base64.b64encode(buffered.getvalue())
    img_str = img_str.decode("ascii")

    # Construct the URL
    image_upload_url = "".join([
        "https://api.roboflow.com/dataset/", DATASET_NAME, "/upload",
        "?api_key=", ROBOFLOW_API_KEY,
        "&name=rabbit.jpg",
        "&split=train"
    ])

    r = requests.post(image_upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    })
    print(r.json())
    return r.json()['success'], r.json()['id']


def uploadAnnotation(imageId, apiResponse):
    # CreateML Dataset Format
    data = []
    annotations = []
    for prediction in apiResponse:
        annotations.append({"label": prediction['class'],
                            "coordinates": {
                                "x": prediction['x'],
                                "y": prediction['y'],
                                "width": prediction['width'],
                                "height": prediction['height']
                            }})
    data.append({
        "image": "rabbit.jpg",
        "annotations": annotations
    })

    # Save to Json File
    with open('activeLearning.json', 'w') as outfile:
        json.dump(data, outfile)

    annotationFilename = "activeLearning.json"

    # Read Annotation as String
    annotationStr = open(annotationFilename, "r").read()

    # Construct the URL
    annotation_upload_url = "".join([
        "https://api.roboflow.com/dataset/", DATASET_NAME, "/annotate/", imageId,
        "?api_key=", ROBOFLOW_API_KEY,
        "&name=", annotationFilename
    ])

    # POST to the API
    r = requests.post(annotation_upload_url, data=annotationStr, headers={
        "Content-Type": "text/plain"
    })

    return r.json()['success']


if __name__ == '__main__':
    # Main loop; infers sequentially until you press "q"
    while True:
        # On "q" keypress, exit
        if (cv2.waitKey(1) == ord('q')):
            break

        # Synchronously get a prediction from the Roboflow Infer API
        image = infer()
        # And display the inference results
        cv2.imshow('image', image)
    # Release resources when finished
    video.release()
    cv2.destroyAllWindows()
