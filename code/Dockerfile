FROM tensorflow/tensorflow:2.3.1-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# Install required python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python

RUN mkdir /work
COPY . /work
WORKDIR /work
