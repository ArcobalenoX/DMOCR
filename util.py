import cv2
import numpy as np
from PIL import Image
import time

import cfg
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow.keras.backend as ktf
from tensorflow.keras.applications.imagenet_utils import preprocess_input

from location.preprocess import resize_image
from location.losses import quad_loss,quad_norm,smooth_l1_loss
from location.nms import nms,rec_region_merge,region_group,region_neighbor,should_merge
from location.predict import crop_rectangle

def location(model, img_path, pixel_threshold):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.image_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    preprocess_input(img,mode='tf')
    x = np.expand_dims(img, axis=0)
    y = model.predict(x)
    y = np.squeeze(y, axis=0)
    y[:, :, :3] = tf.nn.sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    results = []
    with Image.open(img_path) as im:
        d_wight, d_height = resize_image(im, cfg.image_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):
            if np.amin(score) > 0:
                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                #im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
                im = crop_rectangle(im, rescaled_geo)
                results.append(im)
    return results

def recognition(model, img):
    img_size = img.shape
    if (img_size[1] / img_size[0] * 1.0) < 6:
        img_reshape = cv2.resize(img, (int(31.0 / img_size[0] * img_size[1]), cfg.height))

        mat_ori = np.zeros((cfg.height, cfg.width - int(31.0 / img_size[0] * img_size[1]), 3), dtype=np.uint8)
        out_img = np.concatenate([img_reshape, mat_ori], axis=1).transpose([1, 0, 2])
    else:
        out_img = cv2.resize(img, (cfg.width, cfg.height), interpolation=cv2.INTER_CUBIC)
        out_img = np.asarray(out_img)
        out_img = out_img.transpose([1, 0, 2])

    y_pred = model.predict(np.expand_dims(out_img, axis=0))
    shape = y_pred[:, 2:, :].shape
    ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
    result = ''.join([cfg.characters[k] for k in out[0]])
    return result

def loc_and_rec(location_model,recognition_model,img):
    #img = cv2.imread(img)
    img = image.load_img(img)
    img = cv2.resize(img,(cfg.image_size, cfg.image_size))
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    st=time.time()
    #x = preprocess_input(img,mode='tf')
    x = x/128 - 1
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
            
            imcrop = crop_rectangle(img, rescaled_geo)
            results.append(imcrop)
            out_img = cv2.resize(imcrop, (cfg.width, cfg.height)) # (W,H,C)
            out_img.transpose([1, 0, 2]) # (H, W, C)
            y_pred = recognition_model.predict(np.expand_dims(out_img, axis=0))
            shape = y_pred[:, 2:, :].shape
            ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
            out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
            #re_text = ''.join([cfg.characters[k] for k in out[0]])
            re_text = u""
            for c in out[0]:
                if(c != -1):
                    re_text += cfg.characters[c]
                    
            print(re_text)
            cv2.putText(img,'value:'+re_text,org=(100,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                        color=(255,0,0),thickness=2,lineType=cv2.LINE_AA) 
            
            ##四边形描框
            darwgeo=rescaled_geo.astype(np.int32)
            img=cv2.polylines(img,[darwgeo],True,(255,0,0),2)

    et=time.time()
    cv2.putText(img,'FPS:'+'%.1f' % (1/(et-st)),org=(300,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                color=(0,255,0),thickness=2,lineType=cv2.LINE_AA) 

    return img




