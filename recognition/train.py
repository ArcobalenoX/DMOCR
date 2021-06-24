import os

import tensorflow.keras.backend as ktf
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, TensorBoard,ReduceLROnPlateau)
from tensorflow.keras.optimizers import SGD, Adam, Nadam, RMSprop

import cfg
from dataloader import *
from network import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_model, infer_model = CRNN(cfg.height, cfg.width, cfg.label_len, cfg.characters).network()


train_model.load_weights(cfg.saved_model_weights_path)
#train_model.load_weights('saved_model\CRNN_weights.h5')
SGDopt = SGD(lr=0.01, decay=5*1e-5, momentum=0.9, nesterov=True, clipnorm=5)
train_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=SGDopt)

# infer_model.summary()
train_model.summary()
# time_steps为LSTM前的Dense层（None，m，n）中的m
train_model.fit(img_gen_train(batch_size=10, time_steps=16),
                steps_per_epoch=100, initial_epoch=39, epochs=50,
                callbacks=[ModelCheckpoint(cfg.checkpoint_path, monitor='loss', save_best_only=True, save_weights_only=True),
                           TensorBoard('logs'),
                           EarlyStopping(monitor='loss', patience=5),
                           ReduceLROnPlateau(monitor='loss', factor=0.9, patience=5)
                           ]
                )

infer_model.save_weights(cfg.saved_model_weights_path)
