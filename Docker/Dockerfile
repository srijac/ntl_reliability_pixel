FROM python:3.9.10-buster

WORKDIR /app

COPY rclone-v1.57.0-linux-amd64.deb rclone_install.deb
RUN dpkg -i rclone_install.deb

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .