#!bin/sh

docker build -t gpu_competition/pytorch:1.0 .
docker run -it gpu_competition/pytorch