# coding:utf-8
import glob
import os
import socket
import sys
import threading
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as ktf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image

import cfg
from location.network import *
from recognition.network import *
from util import *

location_model = Location().location_network()
location_model.load_weights(cfg.location_weights)
# location_model.summary()

_, recognition_model = CRNN(cfg.height, cfg.width,
                            cfg.label_len, cfg.characters).network()
recognition_model.load_weights(cfg.recognition_weights)
# recognition_model.summary()


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)


def useCamera():
    # 获取摄像头

    capture = cv2.VideoCapture(1)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cv2.namedWindow("window")

    i = 0
    while capture.isOpened():

        # 摄像头打开，读取图像
        flag, image = capture.read()
        # print(image.shape)
        cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
        image = image[:, :480, :]
        # print(image.shape)

        image = loc_and_rec(location_model, recognition_model, image)
        k = cv2.waitKey(1)

        if k == ord("q") or k == ord("Q"):
            break

        elif k == ord('s') or k == ord("S"):
            image = cv2.resize(image, (512, 512),
                               interpolation=cv2.INTER_CUBIC)
            capture_path = f'cap/DM1_{i:03d}.jpg'
            cv2.imwrite(capture_path, image)
            i += 1

        elif k == ord('c'):
            img_path = 'manual.jpg'
            image = cv2.resize(image, (512, 512))
            cv2.imwrite(img_path, image)
            st = time.time()
            ims_re = location(location_model, img_path, cfg.pixel_threshold)
            if(len(ims_re) > 0):
                # print(len(ims_re))
                re_text = recognition(recognition_model, ims_re[0])
                cv2.putText(image, re_text, org=(100, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                #print('recognize result: '+ re_text)
            end = time.time()
            cv2.putText(image, 'FPS:'+'%.1f' % (1/(end-st)), org=(200, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
            cap_path = 'cap/DM1_%3d.jpg' % i
            cv2.imwrite(cap_path, image)
            i += 1

        cv2.imshow("window", image)

    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('camera over')


if __name__ == '__main__':
    useCamera()
