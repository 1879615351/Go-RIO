#!/bin/bash
xhost +

docker build --force-rm -f Dockerfile -t wooseong0929/go-rio:latest .
