FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY lane_detection.py .
RUN pip install opencv-python numpy pyzmq pandas paho-mqtt
CMD ["python3", "lane_detection.py"]