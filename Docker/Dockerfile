FROM nvidia/cuda:11.4.3-cudnn8-devel-ubuntu20.04

WORKDIR /app

COPY rclone-v1.57.0-linux-amd64.deb rclone_install.deb
RUN dpkg -i rclone_install.deb

RUN apt-get update && apt-get install -y python3-pip

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .