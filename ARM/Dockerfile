#ARG  BASE_IMAGE=nvidia/cuda:9.2-runtime-ubuntu16.04
#FROM $BASE_IMAGE
FROM ubuntu:16.04

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

ADD ./app /app/

ENTRYPOINT ["python", "/app/main.py"]