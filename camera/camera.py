import cv2
# import zmq
import numpy as np
import paho.mqtt.client as mqtt
import time
import pickle
import math

# context = zmq.Context()
# socket = context.socket(zmq.PUB)
# socket.bind("tcp://*:5555")  # Broadcasting frames

BROKER = "localhost"
TOPIC = "camera/frame"
OBJECT_DETECTION_TOPIC = "object_detection/frame"
LANE_DETECTED_IMAGE_TOPIC = "lane_detection/frame"

object_detections = []
lane_detected_image = None

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883, 60)

# Subscribers for processed images
# object_receiver = context.socket(zmq.SUB)
# object_receiver.connect("tcp://object_detection:5556")
# object_receiver.setsockopt_string(zmq.SUBSCRIBE, "")

# lane_receiver = context.socket(zmq.SUB)
# lane_receiver.connect("tcp://lane_detection:5557")
# lane_receiver.setsockopt_string(zmq.SUBSCRIBE, "")

def initialize_camera(mode):
    if mode == 1:  # Webcam
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # Width
        cap.set(4, 480)  # Height
    elif mode == 2:  # Phone Camera
        cap = cv2.VideoCapture("http://192.168.233.210:8080/video")
    elif mode == 3:  # ESP32 camera
        cap = cv2.VideoCapture("http://192.168.233.68:81/stream")
    elif mode == 4:  # ESP32 camera
        cap = cv2.VideoCapture('lane2.mp4')
    return cap

# camera_sources = {
#     "usb": 0,
#     "esp32": "http://esp32-cam-ip/stream",
#     "ip": "rtsp://your-ip-camera-stream",
#     "video": "video.mp4"
# }

current_camera = 4
cap = initialize_camera(current_camera)

# # Select camera source
# source = camera_sources["usb"]
# cap = cv2.VideoCapture(source)

# # SOME/IP Setup
# client = someip.("CameraService")

# def on_object_message(client, userdata, message):
#     global object_frame
#     frame_data = np.frombuffer(message.payload, dtype=np.uint8)
#     object_frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

def on_object_message(client, userdata, message):
    global object_detections
    object_detections = pickle.loads(message.payload)

def on_lane_message(client, userdata, message):
    global lane_detected_image
    frame_data = np.frombuffer(message.payload, dtype=np.uint8)
    lane_detected_image = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)

# client = mqtt.Client()
client.message_callback_add(OBJECT_DETECTION_TOPIC, on_object_message)
client.message_callback_add(LANE_DETECTED_IMAGE_TOPIC, on_lane_message)
client.connect(BROKER, 1883, 60)
client.subscribe([(OBJECT_DETECTION_TOPIC, 0), (LANE_DETECTED_IMAGE_TOPIC, 0)])
client.loop_start()

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


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # if current_camera == 4 :
    #     cv2.waitKey(300)
    # # Encode frame
    _, buffer = cv2.imencode(".jpg", frame)
    client.publish(TOPIC, buffer.tobytes())  # Send frame as bytes
    time.sleep(0.1)  # Adjust based on performance needs
    # frame_bytes = buffer.tobytes()

    # # Send frame data over SOME/IP
    # socket.send(frame_bytes)

    # Get object detection results
    # object_bytes = object_receiver.recv()
    # object_nparr = np.frombuffer(object_bytes, np.uint8)
    # objects_frame = cv2.imdecode(object_nparr, cv2.IMREAD_GRAYSCALE)
    
    # # Get lane detection results
    # lane_bytes = lane_receiver.recv()
    # lane_nparr = np.frombuffer(lane_bytes, np.uint8)
    # lanes_frame = cv2.imdecode(lane_nparr, cv2.IMREAD_GRAYSCALE)
    
    # if object_frame is not None and lane_frame is not None:
    #     combined = cv2.addWeighted(object_frame, 1.0, lane_frame, 1.0, 0)
    #     #combined = lane_frame
    #     cv2.imshow("Final Output", combined)

    # combined = cv2.addWeighted(objects_frame, 0.7, lanes_frame, 0.3, 0)

    # cv2.imshow("Final Output", combined)
    
    if lane_detected_image is None:
        print("Waiting for lane-detected image...")
        continue  # Wait until lane image is available

    final_image = lane_detected_image.copy()

    for (x1, y1, x2, y2, conf, class_name) in object_detections:
            cv2.rectangle(final_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Display text
            cv2.putText(final_image, f"{class_name} {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            # cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green boxes

        # Display final output
    cv2.imshow("Final Output", final_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
        if key == ord('1'):
            new_mode = 1 
        elif key == ord('2'):
            new_mode = 2
        elif key == ord('3'):
            new_mode = 3
        elif key == ord('4'):
            new_mode = 4
            
        if new_mode != current_camera:
            cap.release()
            cap = initialize_camera(new_mode)
            current_camera = new_mode
            print(f"Switched to camera {new_mode}")

cap.release()
cv2.destroyAllWindows()
client.disconnect()
