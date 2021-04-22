#!bin/sh

docker build -t gpu_competition/pytorch:1.0 .
docker run -it --gpus all gpu_competition/pytorch:1.0 /bin/bash