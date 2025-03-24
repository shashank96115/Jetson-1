FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY camera.py .
COPY lane2.mp4 .
RUN pip install opencv-python numpy pyzmq
CMD ["python3", "camera.py"]
