FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY camera.py .
COPY british_highway_traffic.mp4 .
COPY test_video.mp4 .
RUN pip install opencv-python numpy pyzmq paho-mqtt
CMD ["python3", "camera.py"]