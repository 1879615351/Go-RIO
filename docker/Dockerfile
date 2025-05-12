FROM osrf/ros:noetic-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV nproc=8

RUN apt update && apt upgrade -y

RUN apt install -y \
    build-essential  \
    cmake \
    pkg-config  \
    htop  \
    gedit  \
    wget \
    git \
    unzip  \
    curl \
    vim \
    gedit \
    software-properties-common \
    libboost-all-dev \
    net-tools \
    iputils-ping \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    ros-noetic-geodesy \
    ros-noetic-nmea-msgs \
    libqglviewer-dev-qt5 \
    && rm -rf /var/lib/apt/lists/*
    
# ceres
WORKDIR /opt
RUN wget -O ceres.tar.gz http://ceres-solver.org/ceres-solver-2.1.0.tar.gz
RUN tar -xzvf ceres.tar.gz -C /opt && rm ceres.tar.gz
WORKDIR /opt/ceres-solver-2.1.0
RUN mkdir build && cd build && cmake .. && make -j$(nproc) && make install
RUN rm -rf /opt/ceres-solver-2.1.0

WORKDIR /opt
RUN git clone https://github.com/RainerKuemmerle/g2o.git /opt/g2o
WORKDIR /opt/g2o
RUN mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc) && make install
RUN rm -rf /opt/g2o

WORKDIR /root/catkin_ws
