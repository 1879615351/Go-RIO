#!/bin/bash
xhost +

export HOST_SHARED_DIR=/home/$USER/gorio_ws
export DATA_DIR=/media/$USER/ws_ssdP

docker run --gpus all --rm -it --ipc=host --net=host --privileged \
    --env="DISPLAY" \
    --gpus=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOST_SHARED_DIR:/root/catkin_ws \
    -v $DATA_DIR:/root/data \
    wooseong0929/go-rio:latest \
    /bin/bash
	
xhost -
