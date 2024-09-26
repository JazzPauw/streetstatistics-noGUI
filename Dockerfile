FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /app
RUN mkdir -p /app/UltralyticsConfig

COPY model.py /app/
COPY short.py /app/
COPY yolov9e-seg.pt /app/

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    pip3 install --no-cache-dir ultralytics opencv-python-headless numpy filterpy \
    networkx==3.1 \
    dill==0.3.8 \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu117

EXPOSE 8000
CMD ["python3", "model.py"]
