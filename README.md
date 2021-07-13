# Computer Vision-Based Rabbit Deterrence System

## Project Overview

This is a computer vision based rabbit deterrence system that leverages the Raspberry Pi and Roboflow to detect and scare away rabbits. How it essentially works is the Raspberry Pi has an object detection model trained with Roboflow Train hosted on it and when it detects a Rabbit, the Bluetooth speaker will play a sound such as a baby crying or a car honking which is meant to scare off the rabbit.

## Special Features of the System

This system implements a couple of different features including:

- Trained Model Using Roboflow Train (which is based on YOLOv5, a state-of-the-art object detection model)
- Uses the MakerHawk External Power Supply (allows for ease of portability)
- Leverages active learning via the Roboflow Upload API
- Includes a Flask Web Server, allowing the user to view the detections remotely
- Integrates a Bluetooth Speaker to scare off the Rabbits

## Materials/Software Used

**Hardware:**

- A Raspberry Pi
- [External Power Supply](https://www.amazon.com/MakerHawk-Raspberry-Uninterruptible-Management-Expansion/dp/B082CVWH3R/ref=sr_1_6?crid=3LJGHA055O4VL&dchild=1&keywords=battery+for+raspberry+pi&qid=1623698007&sprefix=battery+for+raspbe%2Caps%2C184&sr=8-6)
- [Camera for the Pi](https://www.amazon.com/Arducam-Megapixels-Sensor-OV5647-Raspberry/dp/B012V1HEP4/ref=sr_1_6?dchild=1&keywords=Raspberry+Pi+camera&qid=1624689746&sr=8-6)
- [Bluetooth Speaker](https://www.amazon.com/AUDIOVOX-SP881BL-Portable-Bluetooth-Rechargeable/dp/B07F8N6KJ9/ref=sr_1_4?crid=2363N4JZD3B18&dchild=1&keywords=canz+bluetooth+speaker&qid=1626056945&sprefix=CANZ+bluetoot%2Caps%2C173&sr=8-4)

**Software:**

- [Public Dataset](https://public.roboflow.com/object-detection/eastern-cottontail-rabbits)
- [Roboflow Annotate](https://docs.roboflow.com/annotate)
- [Roboflow Train](https://docs.roboflow.com/train)
- [Roboflow Inference API](https://docs.roboflow.com/inference)

## Further Reading

- Blog: [https://blog.roboflow.com/rabbit-deterrence-system/](https://blog.roboflow.com/rabbit-deterrence-system/)
- Video Documentation: [https://www.youtube.com/watch?v=oPvqKgq3ppc](https://www.youtube.com/watch?v=oPvqKgq3ppc)
