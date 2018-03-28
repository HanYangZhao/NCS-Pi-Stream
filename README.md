# Overview
A Python program that host a webserver to stream video from the Pi Camera with Intel's Neural Compute stick. 
The NCS runs mobilenets on Caffe which enables object detection.

![Alt Text](https://i.imgflip.com/2714hl.gif)

# Pre-reqs
* Install the lastest version of Raspbian
* Install the NCSDK : https://github.com/movidius/ncsdk
* Install open cv using the installer in the SDK (this may take a long time)

# Installation
* Clone the repo
* python3 streamer_ncs.py
* Open http://YOUR_PI_IP:8080/cam.mjpg


# Acknowlegement

* https://qiita.com/PINTO/items/b97b3334ed452cb555e2
* https://www.pyimagesearch.com/2018/02/19/real-time-object-detection-on-the-raspberry-pi-with-the-movidius-ncs/
