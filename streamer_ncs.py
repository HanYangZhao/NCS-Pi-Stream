import sys
graph_folder="./graph"
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    exit(1)

from mvnc import mvncapi as mvnc
import numpy as np
import cv2
from os import system
import io, time
from os.path import isfile, join
from queue import Queue
from threading import Thread, Event, Lock
import re
from time import sleep
import http
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import argparse


mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("No devices found")
    quit()
print(len(devices))

devHandle   = []
graphHandle = []

with open(join(graph_folder, "mobilenetgraph"), mode="rb") as f:
    graph = f.read()

for devnum in range(len(devices)):
    devHandle.append(mvnc.Device(devices[devnum]))
    devHandle[devnum].OpenDevice()
    graphHandle.append(devHandle[devnum].AllocateGraph(graph))
    graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)

print("\nLoaded Graphs!!!")

cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('/home/pi/SSD_MobileNet/xxxx.mp4')

if cam.isOpened() != True:
    print("Did you do sudo modprobe bcm2835-v4l2")
    quit()

widowWidth = 640
windowHeight = 480
cam.set(cv2.CAP_PROP_FRAME_WIDTH, widowWidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, windowHeight)

lock = Lock()
frameBuffer = []
results = Queue()
img_to_display = Queue()
lastresults = None

LABELS = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

def camThread():   
    global lastresults
    while True:
        pass
        s, img = cam.read()
        img = cv2.flip(img,-1)

        if not s:
            print("Could not get frame")
            return 0

        lock.acquire()
        if len(frameBuffer)>10:
            for i in range(10):
                del frameBuffer[0]
        frameBuffer.append(img)
        lock.release()
        res = None

        if not results.empty():
            res = results.get(False)
            img = overlay_on_image(img, res)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            lastresults = res
            if img_to_display.qsize()>20:
                img_to_display.get()
            img_to_display.put(img)
        else:
            imdraw = overlay_on_image(img, lastresults)
            #imdraw = cv2.cvtColor(imdraw, cv2.COLOR_BGR2RGB)
            h, w = imdraw.shape[:2]
            if img_to_display.qsize()>20:
                img_to_display.get()
            img_to_display.put(imdraw)


def inferencer(results, lock, frameBuffer, handle):
    failure = 0
    sleep(1)
    while failure < 100:

        lock.acquire()
        if len(frameBuffer) == 0:
            lock.release()
            failure += 1
            continue

        img = frameBuffer[-1].copy()
        del frameBuffer[-1]
        failure = 0
        lock.release()

        now = time.time() 
        im = preprocess_image(img)
        handle.LoadTensor(im.astype(np.float16), None)
        out, userobj = handle.GetResult()
        results.put(out)

def preprocess_image(src):
    img = cv2.resize(src, (300, 300))
    img = img - 127.5
    img = img * 0.007843

    return img


def overlay_on_image(display_image, object_info):

    if isinstance(object_info, type(None)):
        return display_image

    num_valid_boxes = int(object_info[0])
    img_cp = display_image.copy()

    if num_valid_boxes > 0:

        for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not np.isfinite(object_info[base_index]) or
                not np.isfinite(object_info[base_index + 1]) or
                not np.isfinite(object_info[base_index + 2]) or
                not np.isfinite(object_info[base_index + 3]) or
                not np.isfinite(object_info[base_index + 4]) or
                not np.isfinite(object_info[base_index + 5]) or
                not np.isfinite(object_info[base_index + 6])):
                continue

            x1 = max(0, int(object_info[base_index + 3] * img_cp.shape[0]))
            y1 = max(0, int(object_info[base_index + 4] * img_cp.shape[1]))
            x2 = min(img_cp.shape[0], int(object_info[base_index + 5] * img_cp.shape[0]))
            y2 = min(img_cp.shape[1], int(object_info[base_index + 6] * img_cp.shape[1]))

            x1_ = str(x1)
            y1_ = str(y1)
            x2_ = str(x2)
            y2_ = str(y2)

            object_info_overlay = object_info[base_index:base_index + 7]

            min_score_percent = 10
            source_image_width = img_cp.shape[1]
            source_image_height = img_cp.shape[0]

            base_index = 0
            class_id = object_info_overlay[base_index + 1]
            percentage = int(object_info_overlay[base_index + 2] * 100)
            if (percentage >= min_score_percent and int(class_id) == 15):
                label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
                box_left = int(object_info_overlay[base_index + 3] * source_image_width)
                box_top = int(object_info_overlay[base_index + 4] * source_image_height)
                box_right = int(object_info_overlay[base_index + 5] * source_image_width)
                box_bottom = int(object_info_overlay[base_index + 6] * source_image_height)

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)

                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = box_left
                label_top = box_top - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1), label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)
    return img_cp

threads = []

class CamHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server):
        img_src = 'http://{}:{}/cam.mjpg'.format(server.server_address[0], server.server_address[1])
        self.html_page = """
            <html>
                <head></head>
                <body>
                    <img src="{}"/>
                </body>
            </html>""".format(img_src)
        self.html_404_page = """
            <html>
                <head></head>
                <body>
                    <h1>NOT FOUND</h1>
                </body>
            </html>"""
        BaseHTTPRequestHandler.__init__(self, request, client_address, server)

    def do_GET(self):
        if self.path.endswith('.mjpg'):
            self.send_response(http.HTTPStatus.OK)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    image_for_result = img_to_display.get(True)

                    retval, jpg = cv2.imencode('.jpg', image_for_result)
                    if not retval:
                        raise RuntimeError('Could not encode img to JPEG')
                    jpg_bytes = jpg.tobytes()
                    self.wfile.write("--jpgboundary\r\n".encode())
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(jpg_bytes))
                    self.end_headers()
                    self.wfile.write(jpg_bytes)
                    #time.sleep(self.server.read_delay)
                except (IOError, ConnectionError):
                    pass
        elif self.path.endswith('.html'):
            self.send_response(http.HTTPStatus.OK)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.html_page.encode())
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.html_404_page.encode())

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    def __init__(self, capture_path, server_address, RequestHandlerClass, bind_and_activate=True):
        HTTPServer.__init__(self, server_address, RequestHandlerClass, bind_and_activate)
        ThreadingMixIn.__init__(self)

        def serve_forever(self, poll_interval=0.1):
            super().serve_forever(poll_interval)

def main():
    for devnum in range(len(devices)):
        t = Thread(target=inferencer, args=(results, lock, frameBuffer, graphHandle[devnum]))
        t.start()
        threads.append(t)
    c_t = Thread(target=camThread)
    c_t.start()
    threads.append(c_t)
    server = ThreadedHTTPServer("empty",('0.0.0.0', 8080), CamHandler)
    print("server started")
    server.serve_forever()

if __name__ == '__main__':
    main()
