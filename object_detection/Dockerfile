FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY object_detection.py .
COPY yolo11n.pt .
RUN pip install opencv-python numpy pyzmq ultralytics
CMD ["python3", "object_detection.py"]
