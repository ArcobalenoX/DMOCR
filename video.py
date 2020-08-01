#coding:utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import socket
import time
import sys

import glob
import threading
import serial
import serial.tools.list_ports

import tensorflow as tf
import tensorflow.keras.backend as ktf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model,Model
from tensorflow.keras.preprocessing import image

from location.network import *
from recognition.network import *
from util import *
import cfg

location_model = Location().location_network()
location_model.load_weights(cfg.location_weights)
#location_model.summary()

_, recognition_model = CRNN(cfg.height, cfg.width,  cfg.label_len, cfg.characters).network()
recognition_model.load_weights(cfg.recognition_weights)
#recognition_model.summary()


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)


def useCamera():
    # 获取摄像头

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,512)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,512)
    cv2.namedWindow("window")

    i=0
    while capture.isOpened():
        
        # 摄像头打开，读取图像
        flag, image = capture.read()
        #print(image.shape)               
        cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
        image=image[:,:480,:]
        #print(image.shape)

        '''
        img = cv2.resize(img,(512,512))
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #ims_re = location(location_model, img_path,cfg.pixel_threshold)
        #d_wight, d_height = resize_image(img, cfg.image_size)
        st=time.time()
        x = preprocess_input(img,mode='tf')
        x = np.expand_dims(x, axis=0)
        y = location_model.predict(x)
        y = np.squeeze(y, axis=0)
        y[:, :, :3] = tf.nn.sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], cfg.pixel_threshold)
        activation_pixels = np.where(cond)
        quad_scores, quad_after_nms = nms(y, activation_pixels)
        results=[]
        for score, geo, s in zip(quad_scores, quad_after_nms,range(len(quad_scores))):
            if np.amin(score) > 0:
                #rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo = geo

                ##字符序列识别（训练时使用BGR格式图片）
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                imcrop = crop_rectangle(img, rescaled_geo)
                results.append(imcrop)
                re_text = recognition(recognition_model, imcrop)
                cv2.putText(img,'value:'+re_text,org=(100,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                            color=(255,0,0),thickness=2,lineType=cv2.LINE_AA) 
                
                ##四边形描框
                darwgeo=rescaled_geo.astype(np.int32)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img=cv2.polylines(img,[darwgeo],True,(255,0,0),2)



        et=time.time()
        cv2.putText(img,'FPS:'+'%.1f' % (1/(et-st)),org=(300,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                   color=(0,255,0),thickness=2,lineType=cv2.LINE_AA) 

        '''
       
        k = cv2.waitKey(1)
        if k == ord("q") or k == ord("Q"):
            break

        elif k == ord('s') or k == ord("S"):
            image = cv2.resize(image,(512,512),interpolation=cv2.INTER_CUBIC)
            capture_path = f'cap/DM1_{i:03d}.jpg'
            cv2.imwrite(capture_path, image)
            i+=1


        elif k == ord('c'):
            img_path ='manual.jpg'           
            image = cv2.resize(image,(512,512))
            cv2.imwrite(img_path,image)
            st=time.time()
            ims_re = location(location_model, img_path,cfg.pixel_threshold)
            if(len(ims_re)>0):
                #print(len(ims_re))
                re_text = recognition(recognition_model, ims_re[0])
                cv2.putText(image,re_text,org=(100,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                        color=(255,0,0),thickness=2,lineType=cv2.LINE_AA) 
                #print('recognize result: '+ re_text)
            end=time.time()
            cv2.putText(image,'FPS:'+'%.1f' % (1/(end-st)),org=(200,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                        color=(0,0,255),thickness=2,lineType=cv2.LINE_AA) 
            cap_path = 'cap/DM1_%3d.jpg'%i
            cv2.imwrite(cap_path,image)
            i+=1


    
        cv2.imshow("window", image)

    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()
    print('camera over')




if __name__ == '__main__':
    #thread= threading.Thread(group=None,target=servo)
    #thread.start()
    useCamera()
    #thread.join()



