FROM nvidia/cuda:10.2-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install -r requirements.txt

WORKDIR /competition

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs