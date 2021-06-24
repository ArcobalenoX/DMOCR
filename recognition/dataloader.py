import os
import cv2
import numpy as np
from tensorflow.keras.callbacks import Callback

import cfg

charset = cfg.characters
#print(charset)
charset_dict = {key: val for val, key in enumerate(charset)}
int_to_string_dict = dict(enumerate(charset))  # {0: 'A', 1: 'B', 2: 'C', 3: 'D'..}
#print(int_to_string_dict)


def string_to_label(string):
    """Convert a string to a list of labels"""
    label = [charset_dict[c] for c in string]
    return label
def label_to_string(labels):
    """Convert a list of labels to the corresoponding string"""
    string = u""
    for c in labels:
        if(c != -1):
            string += charset[c]
            #string = string.join(charset[c] )
    return string


train_imgs = open(cfg.train_txt_path, 'r').readlines()
train_imgs_num = len(train_imgs)
val_imgs = open(cfg.val_txt_path, 'r').readlines()
val_imgs_num = len(val_imgs)

def img_gen_train(batch_size,time_steps):
    imgs = np.zeros((batch_size, cfg.height, cfg.width, 3), dtype=np.uint8)
    labels = np.zeros((batch_size, cfg.label_len), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            while True:
                pick_index = np.random.randint(0, train_imgs_num - 1)
                train_imgs_split = train_imgs[pick_index].split()
                label = train_imgs_split[1]
                img_path = os.path.join(cfg.ocr_dataset_path, train_imgs_split[0])
                
                img = cv2.imread(img_path)      # (H, W, C)
                if (img is not None) and len(label) <= cfg.label_len:   #读取到有效图片
                    break
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB )
            out_img = cv2.resize(img, (cfg.width, cfg.height)) # (W,H,C)
            out_img.transpose([1, 0, 2]) # (H, W, C)

            # due to the explanation of ctc_loss, try to not add "|" for blank
            while len(label) < cfg.label_len:
                label += "|"

            imgs[i] = out_img
            labels[i] = string_to_label(label)
        yield [imgs, labels, np.ones(batch_size)*(time_steps-2), np.ones(batch_size)*cfg.label_len], labels

def img_gen_val(batch_size=100):
    imgs = np.zeros((batch_size, cfg.height, cfg.width, 3), dtype=np.uint8)
    labels = np.zeros((batch_size, cfg.label_len), dtype=np.uint8)
    #labels = []

    while True:
        for i in range(batch_size):

            while True:
                pick_index = np.random.randint(0, val_imgs_num - 1)
                val_imgs_split = [m for m in val_imgs[pick_index].split()]
                label = val_imgs_split[1]

                img_path = cfg.ocr_dataset_path + val_imgs_split[0]
                img = cv2.imread(img_path)  # (H, W, C)

                if (img is not None) and len(label) <= cfg.label_len:   #读取到有效图片
                    break
            out_img = cv2.resize(img, (cfg.width, cfg.height)) # (W,H,C)
            out_img.transpose([1, 0, 2]) # (H, W, C)

            imgs[i] = out_img
            labels[i] = label 
            #labels.append(label)
        yield imgs, labels


class Evaluate(Callback):

    def on_epoch_end(self, epoch, logs=None):

        def evaluate(infer_model):
            correct_prediction = 0
            generator = img_gen_val()
            x_test, y_test = next(generator)
            y_pred = infer_model.predict(x_test)
            shape = y_pred[:, 2:, :].shape
            ctc_decode = ktf.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
            out = ktf.get_value(ctc_decode)[:, :cfg.label_len]

            for m in range(100):
                result_str = label_to_string(out[m])
                result_str = result_str.replace('|', '')
                if result_str == y_test[m]:
                    correct_prediction += 1
                else:
                    print(result_str, y_test[m])

            return correct_prediction * 1.0
    
        acc = evaluate(infer_model)
        print('')
        print('acc:'+str(acc)+"%")