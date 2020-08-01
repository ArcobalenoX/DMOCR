import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from network import Location
from nms import nms
from preprocess import resize_image
import cfg

def crop_rectangle(img, geo):
    rect = cv2.minAreaRect(geo.astype(int))
    center, size, angle = rect[0], rect[1], rect[2]
    if(angle > -45):
        center = tuple(map(int, center))
        size = tuple([int(rect[1][0] + 10), int(rect[1][1] + 10)])
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    else:
        center = tuple(map(int, center))
        size = tuple([int(rect[1][1] + 10), int(rect[1][0]) + 10])
        angle -= 270
        height, width = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D(center, angle, 1)
        img_rot = cv2.warpAffine(img, M, (width, height))
        img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop


def predict(east_detect, img_path, pixel_threshold, quiet=True):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    preprocess_input(img,mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = tf.math.sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    try:
        with Image.open(img_path) as im:
            im_array = image.img_to_array(im.convert('RGB'))
            d_wight, d_height = resize_image(im, cfg.image_size)
            scale_ratio_w = d_wight / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            quad_im = im.copy()
            quad_draw = ImageDraw.Draw(quad_im)
            txt_items = []
            flag = False
            for score, geo, s in zip(quad_scores, quad_after_nms,range(len(quad_scores))):
                if np.amin(score) > 0:
                    flag = True
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='blue')
                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                    txt_item = ','.join(map(str, rescaled_geo_list))
                    #print(rescaled_geo_list)
                    txt_items.append(txt_item + '\n')
                    if cfg.detection_box_crop:
                        img_crop = crop_rectangle(im_array, rescaled_geo)
                        if not os.path.exists('output_crop'):
                            os.mkdir('output_crop')
                        cv2.imwrite(os.path.join('output_crop', img_path.split('/')[-1].split('.')[0] + '.jpg'), img_crop)
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            if flag:
                if not os.path.exists('output'):
                    os.mkdir('output')
                quad_im.save(os.path.join('output', img_path.split('/')[-1].split('.')[0] + '_predict.jpg'))
            if cfg.predict_write2txt and len(txt_items) > 0:
                if not os.path.exists('output_txt'):
                    os.mkdir('output_txt')
                with open(os.path.join("output_txt", img_path.split('/')[-1].split('.')[0] + '.txt'), 'w') as f_txt:
                    f_txt.writelines(txt_items)
    except:
        print(img_path+'  open error')


if __name__ == '__main__':
   
    location_model = Location().location_network()
    location_model.summary()
    location_model.load_weights(cfg.saved_model_weights_path)
    #location_model.load_weights('saved_model/weights_E15_L0.10200.h5')

    test_all=True
    if test_all:
        img_list = os.listdir(r"test_imgs/")
        for img_path in img_list:
            predict(location_model, os.path.join(r"test_imgs/", img_path),cfg.pixel_threshold)
            #print(img_path)
    else:
        predict(location_model, 'test_imgs/002.jpg', cfg.pixel_threshold)
