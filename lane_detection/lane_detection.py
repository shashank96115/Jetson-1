import cv2
import zmq
import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt
import time


# context = zmq.Context()
# receiver = context.socket(zmq.SUB)
# receiver.connect("tcp://camera:5555")  # Subscribe to camera
# receiver.setsockopt_string(zmq.SUBSCRIBE, "")

# publisher = context.socket(zmq.PUB)
# publisher.bind("tcp://*:5557")  # Publish lane detection results

# def detect_lanes(frame):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray, 50, 150)
#     return edges

BROKER = "localhost"
CAMERA_TOPIC = "camera/frame"
LANE_DETECTION_TOPIC = "lane_detection/frame"

def region_selection(image):
	"""
	Determine and cut the region of interest in the input image.
	Parameters:
		image: we pass here the output from canny where we have 
		identified edges in the frame
	"""
	# create an array of the same size as of the input image 
	mask = np.zeros_like(image) 
	# if you pass an image with more then one channel
	if len(image.shape) > 2:
		channel_count = image.shape[2]
		ignore_mask_color = (255,) * channel_count
	# our image only has one channel so it will go under "else"
	else:
		# color of the mask polygon (white)
		ignore_mask_color = 255
	# creating a polygon to focus only on the road in the picture
	# we have created this polygon in accordance to how the camera was placed
	rows, cols = image.shape[:2]
	bottom_left = [cols * 0.1, rows * 0.95]
	top_left	 = [cols * 0.4, rows * 0.6]
	bottom_right = [cols * 0.9, rows * 0.95]
	top_right = [cols * 0.6, rows * 0.6]
	vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
	# filling the polygon with white color and generating the final mask
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	# performing Bitwise AND on the input image and mask to get only the edges on the road
	masked_image = cv2.bitwise_and(image, mask)
	return masked_image

def hough_transform(image):
	"""
	Determine and cut the region of interest in the input image.
	Parameter:
		image: grayscale image which should be an output from the edge detector
	"""
	# Distance resolution of the accumulator in pixels.
	rho = 1			
	# Angle resolution of the accumulator in radians.
	theta = np.pi/180
	# Only lines that are greater than threshold will be returned.
	threshold = 20	
	# Line segments shorter than that are rejected.
	minLineLength = 20
	# Maximum allowed gap between points on the same line to link them
	maxLineGap = 500	
	# function returns an array containing dimensions of straight lines 
	# appearing in the input image
	return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
						minLineLength = minLineLength, maxLineGap = maxLineGap)
	
def average_slope_intercept(lines):
	"""
	Find the slope and intercept of the left and right lanes of each image.
	Parameters:
		lines: output from Hough Transform
	"""
	left_lines = [] #(slope, intercept)
	left_weights = [] #(length,)
	right_lines = [] #(slope, intercept)
	right_weights = [] #(length,)
	
	for line in lines:
		for x1, y1, x2, y2 in line:
			if x1 == x2:
				continue
			slope = (y2 - y1) / (x2 - x1)
            # Filter out near-horizontal lines (adjust threshold as needed)
			if abs(slope) < 0.5:
				continue  # Skip lines with near-zero slopes
			intercept = y1 - (slope * x1)
			length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
			if slope < 0:
				left_lines.append((slope, intercept))
				left_weights.append(length)
			else:
				right_lines.append((slope, intercept))
				right_weights.append(length)
	# 
	left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
	right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
	return left_lane, right_lane

def pixel_points(y1, y2, line):
	"""
	Converts the slope and intercept of each line into pixel points.
		Parameters:
			y1: y-value of the line's starting point.
			y2: y-value of the line's end point.
			line: The slope and intercept of the line.
	"""
	if line is None:
		return None
	slope, intercept = line
	# x1 = int((y1 - intercept)/slope)
	# x2 = int((y2 - intercept)/slope)
	# y1 = int(y1)
	# y2 = int(y2)
	epsilon = 1e-8  # Threshold to check for near-zero slope
	if abs(slope) < epsilon:
		return None  # Skip horizontal lines
	try:
		x1 = int((y1 - intercept) / slope)
		x2 = int((y2 - intercept) / slope)
	except ZeroDivisionError:
		return None
	y1 = int(y1)
	y2 = int(y2)
	return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
	"""
	Create full lenght lines from pixel points.
		Parameters:
			image: The input test image.
			lines: The output lines from Hough Transform.
	"""
	left_lane, right_lane = average_slope_intercept(lines)
	y1 = image.shape[0]
	y2 = y1 * 0.6
	left_line = pixel_points(y1, y2, left_lane)
	right_line = pixel_points(y1, y2, right_lane)
	return left_line, right_line

	
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
	"""
	Draw lines onto the input image.
		Parameters:
			image: The input test image (video frame in our case).
			lines: The output lines from Hough Transform.
			color (Default = red): Line color.
			thickness (Default = 12): Line thickness. 
	"""
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

def frame_processor(image):
	"""
	Process the input frame to detect lane lines.
	Parameters:
		image: image of a road where one wants to detect lane lines
		(we will be passing frames of video to this function)
	"""
	# convert the RGB image to Gray scale
	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# applying gaussian Blur which removes noise from the image 
	# and focuses on our region of interest
	# size of gaussian kernel
	kernel_size = 5
	# Applying gaussian blur to remove noise from the frames
	blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
	# first threshold for the hysteresis procedure
	low_t = 50
	# second threshold for the hysteresis procedure 
	high_t = 150
	# applying canny edge detection and save edges in a variable
	edges = cv2.Canny(blur, low_t, high_t)
	# since we are getting too many edges from our image, we apply 
	# a mask polygon to only focus on the road
	# Will explain Region selection in detail in further steps
	region = region_selection(edges)
	# Applying hough transform to get straight lines from our image 
	# and find the lane lines
	# Will explain Hough Transform in detail in further steps
	hough = hough_transform(region)
	#lastly we draw the lines on our resulting frame and return it as output 
	result = draw_lane_lines(image, lane_lines(image, hough))
	return result

# driver function
# def process_video(test_video, output_video):
# 	"""
# 	Read input video stream and produce a video file with detected lane lines.
# 	Parameters:
# 		test_video: location of input video file
# 		output_video: location where output video file is to be saved
# 	"""
# 	# read the video file using VideoFileClip without audio
# 	input_video = editor.VideoFileClip(test_video, audio=False)
# 	# apply the function "frame_processor" to each frame of the video
# 	# will give more detail about "frame_processor" in further steps
# 	# "processed" stores the output video
# 	processed = input_video.fl_image(frame_processor)
# 	# save the output video stream to an mp4 file
# 	processed.write_videofile(output_video, audio=False)
# 	# input_video.write_videofile(output_video, audio=False)

# def initialize_camera(mode):
#     if mode == 1:  # Webcam
#         cap = cv2.VideoCapture(0)
#         # cap.set(3, 640)  # Width
#         # cap.set(4, 480)  # Height
#     elif mode == 2:  # Phone Camera
#         cap = cv2.VideoCapture("http://192.168.233.210:8080/video")
#     elif mode == 3:  # ESP32 camera
#         cap = cv2.VideoCapture("http://192.168.233.68:81/stream")
#     elif mode == 4:  # ESP32 camera
#         cap = cv2.VideoCapture('test_video.mp4')
#     return cap	

# current_camera = 4
# cap = initialize_camera(current_camera)

def on_message(client, userdata, message):
    frame_data = np.frombuffer(message.payload, dtype=np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR) 

    # Placeholder: Run lane detection (replace with actual model)
    detected_frame = frame_processor(frame)

    _, buffer = cv2.imencode('.jpg', detected_frame)
    client.publish(LANE_DETECTION_TOPIC, buffer.tobytes())  # Publish result

# Initialize MQTT
client = mqtt.Client()
client.on_message = on_message
client.connect(BROKER, 1883, 60)
client.subscribe(CAMERA_TOPIC)
client.loop_forever()

# while True:
#     # Read frame from camera
#     # ret, frame = cap.read()
#     # if not ret:
#     #     print("Failed to capture frame")
#     #     break
	
#     frame_bytes = receiver.recv()
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Process frame for lane detection
#     processed_frame = frame_processor(frame)
	
#     _, buffer = cv2.imencode(".jpg", processed_frame)

#     publisher.send(buffer.tobytes())
    
    # Display processed frame
    # cv2.imshow('Lane Detection', processed_frame)
	
#     # Handle keyboard input
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
#         if key == ord('1'):
#             new_mode = 1 
#         elif key == ord('2'):
#             new_mode = 2
#         elif key == ord('3'):
#             new_mode = 3
#         elif key == ord('4'):
#             new_mode = 4
        
#         if new_mode != current_camera:
#             cap.release()
#             cap = initialize_camera(new_mode)
#             current_camera = new_mode
#             print(f"Switched to camera {new_mode}")

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# calling driver function
#process_video('test_video.mp4','output.mp4')
# img = cv2.imread('lane2.jpg')
# res=frame_processor(img)
# cv2.imwrite('lane_out.jpg', res)

# while True:
#     frame_bytes = receiver.recv()
#     nparr = np.frombuffer(frame_bytes, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     lane_edges = detect_lanes(frame)
#     _, buffer = cv2.imencode(".jpg", lane_edges)

#     publisher.send(buffer.tobytes())
