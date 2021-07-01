import base64
import io
import os

import cv2
import requests
import pygame as pg
from PIL import Image

from config import *
from threading import Thread

import copy


class Camera:
    def __init__(self, source):
        # Video capture
        self.video = cv2.VideoCapture(source)
        self.soundFile = os.getcwd() + os.sep + 'car-honk.mp3'
        self.soundCondition = False
        self.uploadCondition = False
        pg.mixer.init()
        pg.mixer.music.load(self.soundFile)

    def getRawFrame(self):
        # Returns the raw frame
        _, frameToReturn = self.video.read()
        return frameToReturn

    # Frame with annotations
    def getFrameAnnotations(self):
        success, img = self.video.read()
        # Rotate Camera Upside down if needed
        # img = cv2.rotate(img, cv2.ROTATE_180)
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

        rawImg = copy.deepcopy(img)
        # Draw all predictions
        respCount = 0
        for prediction in resp:
            if prediction["confidence"] > CONFIDENCE_THRESHOLD:
                respCount += 1
                self.writeOnStream(prediction['x'], prediction['y'], prediction['width'], prediction['height'],
                                   prediction['class'],
                                   img)

        return respCount > 0, img, rawImg, resp

    def getFrame(self):
        sound, img, rawImg, apiResponse = self.getFrameAnnotations()
        # Multithread sound
        if not self.soundCondition and sound:
            self.soundCondition = True
            soundThread = Thread(target=self.playSound)
            soundThread.start()

        # Multithread Active Learning
        if not self.uploadCondition and sound:
            # Do not add blurry images to dataset
            if cv2.Laplacian(rawImg, cv2.CV_64F).var() > LAPLACIAN_THRESHOLD:
                self.uploadCondition = True
                uploadThread = Thread(target=self.activeLearning, args=[rawImg, apiResponse])
                uploadThread.start()

        return img

    def activeLearning(self, image, apiResponse):
        success, imageId = self.uploadImage(image)
        if success:
            self.uploadAnnotation(imageId, apiResponse)

        self.uploadCondition = False

    def playSound(self):
        '''
        stream music with mixer.music module in blocking manner
        this will stream the sound from disk while playing
        '''
        clock = pg.time.Clock()
        pg.mixer.music.play()

        while pg.mixer.music.get_busy():
            clock.tick(30)
        self.soundCondition = False

    def uploadImage(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilImage = Image.fromarray(frame)
        # Convert to JPEG Buffer
        buffered = io.BytesIO()
        pilImage.save(buffered, quality=90, format="JPEG")

        # Base 64 Encode
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        r = requests.post(image_upload_url, data=img_str, headers={
            "Content-Type": "application/x-www-form-urlencoded"
        })
        return r.json()['success'], r.json()['id']

    def uploadAnnotation(self, imageId, apiResponse):
        # CreateML Dataset Format
        data = []
        annotations = []
        for prediction in apiResponse:
            if prediction["confidence"] < CONFIDENCE_THRESHOLD:
                continue
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
        # return r.json()['success']
        return True

    def writeOnStream(self, x, y, width, height, className, frame):
        # Draw a Rectangle around detected image
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y - height / 2)),
                      (255, 0, 0), 2)

        # Draw filled box for class name
        cv2.rectangle(frame, (int(x - width / 2), int(y + height / 2)), (int(x + width / 2), int(y + height / 2) + 35),
                      (255, 0, 0), cv2.FILLED)

        # Set label font + draw Text
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, className, (int(x - width / 2 + 6), int(y + height / 2 + 26)), font, 0.5, (255, 255, 255), 1)
