import cv2
import numpy as np
import zmq
import time
import queue
from threading import Thread

context = zmq.Context()

# Publisher for raw frames
frame_pub = context.socket(zmq.PUB)
frame_pub.bind("tcp://0.0.0.0:5555")

print("Publisher: Ready to send messages...")

# Subscriber for detected frames
result_sub = context.socket(zmq.SUB)
result_sub.connect("tcp://localhost:5556")
result_sub.setsockopt_string(zmq.SUBSCRIBE, "")
# time.sleep(20) 


def initialize_camera(mode):
    if mode == 1:  # Webcam
        cap = cv2.VideoCapture(0)
    elif mode == 2:  # Phone Camera
        cap = cv2.VideoCapture("http://192.168.233.210:8080/video")
    elif mode == 3:  # ESP32 camera
        cap = cv2.VideoCapture("http://192.168.233.68:81/stream")
    elif mode == 4:  # ESP32 camera
        cap = cv2.VideoCapture('lane2.mp4')

    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    return cap

current_camera = 4
cap = initialize_camera(current_camera)

# def display_thread():
#     while True:
#         try:
#             result = result_sub.recv(flags=zmq.NOBLOCK)
#             buffer = np.frombuffer(result, dtype=np.uint8)
#             img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
#             cv2.imshow('Detections', img)
#             cv2.waitKey(1)
#         except zmq.Again:
#             time.sleep(0.01)

# Thread(target=display_thread, daemon=True).start()
time.sleep(10) 
message = f"Hello from Publisher!"
print(f"Sending: {message}")
frame_pub.send_string(message)
time.sleep(5) 
message = result_sub.recv_string()
print(f"Received: {message}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Encode frame
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    frame_pub.send(buffer.tobytes()) 
    time.sleep(0.1) 

    # result = result_sub.recv(flags=zmq.NOBLOCK)
    # buffer = np.frombuffer(result, dtype=np.uint8)
    # img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    # cv2.imshow('Detections', img)

    # message = "Hello from Publisher!"
    # frame_pub.send_string(message)  # Send message
    # print(f"Sent: {message}")

    try:
        # message = result_sub.recv_string()  # Blocking receive
        # print(f"Received: {message}")
        result = result_sub.recv(flags=zmq.NOBLOCK)
        buffer = np.frombuffer(result, dtype=np.uint8)
        img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        cv2.imshow('Detections', img)
        # cv2.waitKey(1)
    except zmq.Again:
        time.sleep(0.01)
    
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
