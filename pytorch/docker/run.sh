#!/bin/bash
if [ ! -d $USER ]
then
  mkdir -p $USER
fi

xhost +local:
docker run -it --net=host \
  --gpus all \
  --volume=/dev:/dev \
  --name=dockerdeep_learning \
  --workdir=/home/$USER \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=dockerdeep_learning-dev \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$(pwd)/$USER:/home/$USER" \
  -v "$(pwd)/../datasets:/home/datasets" \
  -v "$(pwd)/../BYOL:/home/BYOL" \
  dockerdeep_learning:latest
