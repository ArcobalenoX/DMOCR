
import os
import cv2
import numpy as np
import tensorflow.keras.backend as ktf
from dataloader import *
from network import CRNN

import cfg



def predict(infer_model, img_path):
    img = cv2.imread(img_path)
    print(img_path)
    out_img = cv2.resize(img, (cfg.width, cfg.height)) # (W,H,C)
    out_img = cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB )
    out_img.transpose([1, 0, 2]) # (H, W, C)
    y_pred = infer_model.predict(np.expand_dims(out_img, axis=0))
    shape = y_pred[:, 2:, :].shape
    ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = ktf.get_value(ctc_decode)[:, :cfg.label_len]
    print(out[0])

    result = label_to_string(out[0])

    return result


if __name__ == '__main__':

    _, infer_model = CRNN(cfg.height, cfg.width,  cfg.label_len, cfg.characters).network()
    infer_model.load_weights(cfg.saved_model_weights_path)
    #infer_model.load_weights('saved_model/BLSTM_E103_L0.002.h5')

    # infer_model.summary()
    imgs_path = cfg.ocr_dataset_path
    #imgs_path=r'Z:\Code\Python\datas\meter512\crnn_imgs'
    result_path='result.txt'

    imgs_list = os.listdir(imgs_path)
    result_txt = open(result_path, 'w+')

    for img_name in imgs_list:
        result = predict(infer_model, os.path.join(imgs_path, img_name))
        result = img_name + " " + result + "\n"
        print(result)
        result_txt.write(result)
    result_txt.close()

