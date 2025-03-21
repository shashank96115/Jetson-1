from ultralytics import YOLO
import cv2
# import zmq
import numpy as np
import math
import paho.mqtt.client as mqtt
import time
import pickle

BROKER = "localhost"
CAMERA_TOPIC = "camera/frame"
OBJECT_DETECTION_TOPIC = "object_detection/frame"

# Load YOLOv5 model
# model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
# model.eval()

model = YOLO("./yolo11n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

def on_message(client, userdata, message):
    frame_data = np.frombuffer(message.payload, dtype=np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    img = frame
    results = model(img, stream=True)
    annotations = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # annotations.append((x1, y1, x2 - x1, y2 - y1))
    #         cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence and class
            confidence = math.ceil(box.conf[0].item() * 100) / 100
            cls = int(box.cls[0])
            class_name = classNames[cls] if cls < len(classNames) else "Unknown"
            annotations.append((x1, y1, x2, y2, confidence, class_name))
    #         # Display text
    #         detected_frame = cv2.putText(img, f"{classNames[cls]} {confidence}", (x1, y1-10), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # _, buffer = cv2.imencode('.jpg', detected_frame)
    # client.publish(OBJECT_DETECTION_TOPIC, buffer.tobytes())  # Publish result
    client.publish(OBJECT_DETECTION_TOPIC, pickle.dumps(annotations))

# Initialize MQTT
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.subscribe(CAMERA_TOPIC)
client.loop_forever()

# # ZMQ Setup
# context = zmq.Context()
# receiver = context.socket(zmq.SUB)
# receiver.connect("tcp://camera:5555")  # Subscribe to camera
# receiver.setsockopt_string(zmq.SUBSCRIBE, "")

# publisher = context.socket(zmq.PUB)
# publisher.bind("tcp://*:5556")  # Publish object detection results

# while True:
#     frame_bytes = receiver.recv()
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     img = frame
#     # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     # results = model(image)
#     # detections = results.pandas().xyxy[0].to_dict(orient="records")

#     # Object detection
#     results = model(img, stream=True)

#     # Process results
#     for r in results:
#         boxes = r.boxes
#         for box in boxes:
#             # Bounding box
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # Confidence and class
#             confidence = math.ceil(box.conf[0].item() * 100) / 100
#             cls = int(box.cls[0])
            
#             # Display text
#             cv2.putText(img, f"{classNames[cls]} {confidence}", (x1, y1-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     _, buffer = cv2.imencode(".jpg", img)

#     publisher.send(buffer.tobytes())
#     # publisher.send_json(img)
