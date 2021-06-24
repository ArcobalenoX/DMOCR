#coding=utf-8

import sys
import os
from tensorflow.keras.preprocessing import image
import time
import argparse


from location.network import *
from recognition.network import *
from util import *
import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

location_model = Location().location_network()
location_model.load_weights(cfg.location_weights)

_, recognition_model = CRNN(cfg.height, cfg.width,  cfg.label_len, cfg.characters).network()
recognition_model.load_weights(cfg.recognition_weights)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of LR")
    parser.add_argument('-p', '--path', default='demo/300.jpg')
    args = parser.parse_args()
    img_path=args.path
    ims_re = location(location_model, img_path, cfg.pixel_threshold)
    if (len(ims_re) > 0 ):
        re_text = recognition(recognition_model, ims_re[0])
        result ="\n{} recognize result: {}\n".format(img_path,re_text)
        print(result)

 


