import json

with open('roboflow_config.json') as f:
    config = json.load(f)

    ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
    ROBOFLOW_MODEL = config["ROBOFLOW_MODEL"]
    ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
    LOCAL_SERVER = config["LOCAL_SERVER"]
    DATASET_NAME = config["DATASET_NAME"]
    LAPLACIAN_THRESHOLD = config["LAPLACIAN_THRESHOLD"]
    CONFIDENCE_THRESHOLD = config["CONFIDENCE_THRESHOLD"]

if not LOCAL_SERVER:
    infer_url = "".join([
        "https://detect.roboflow.com/" + ROBOFLOW_MODEL,
        "?api_key=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])
else:
    infer_url = "".join([
        "http://127.0.0.1:9001/" + ROBOFLOW_MODEL,
        "?api_key=" + ROBOFLOW_API_KEY,
        "&name=YOUR_IMAGE.jpg"
    ])

# Construct the URL
image_upload_url = "".join([
    "https://api.roboflow.com/dataset/", DATASET_NAME, "/upload",
    "?api_key=", ROBOFLOW_API_KEY,
    "&name=rabbit.jpg",
    "&split=train"
])
