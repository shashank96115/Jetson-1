from ultralytics import YOLO
import cv2
import zmq
import numpy as np
import math
import time

context = zmq.Context()

# Subscriber for raw frames
frame_sub = context.socket(zmq.SUB)
frame_sub.connect("tcp://localhost:5555")
frame_sub.setsockopt_string(zmq.SUBSCRIBE, "")

# Publisher for detected frames
result_pub = context.socket(zmq.PUB)
result_pub.bind("tcp://0.0.0.0:5556")

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

def process_frame(frame_data):
    # Decode and process frame
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    img = frame.copy()
    results = model(img, stream=True)

    # Process results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Confidence and class
            confidence = math.ceil(box.conf[0].item() * 100) / 100
            cls = int(box.cls[0])
            
            # Display text
            cv2.putText(img, f"{classNames[cls]} {confidence}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
    return img
print("waiting for messge")
message = frame_sub.recv_string()
print("messge received")
print(f"Received: {message}")
message = f"Hello from Subscriber!"
print(f"Sending: {message}")
result_pub.send_string(message)

while True:
    try:
        # message = frame_sub.recv_string()  # Blocking receive
        # print(f"Received: {message}")
        
        # Receive frame
        frame_data = frame_sub.recv(flags=zmq.NOBLOCK)

        
        # Process and send result
        annotated_frame = process_frame(frame_data)
        _, result_buffer = cv2.imencode(".jpg", annotated_frame)
        result_pub.send(result_buffer.tobytes())
        
    except zmq.Again:
        time.sleep(0.01)
