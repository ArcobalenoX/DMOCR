import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam, Nadam
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cfg
from dataloader import gen
from losses import quad_loss
#from network import Location

from experiments.EAST.network import East
from experiments.efficient.network import Location
#from experiments.location.network import Location

#location_model = East().east_network()
location_model = Location().location_network()
location_model.summary()

location_model.compile(loss=quad_loss, optimizer=Adam(lr=0.001))
if os.path.exists(cfg.saved_model_weights_path):
    location_model.load_weights(cfg.saved_model_weights_path)
#location_model.load_weights("saved_model/location_weights.h5")

location_model.fit(gen(),
                   steps_per_epoch=cfg.steps_per_epoch,
                   initial_epoch=12,
                   epochs=25,
                   validation_data=gen(is_val=True),
                   validation_steps=cfg.validation_steps,
                   callbacks=[TensorBoard(cfg.tensorboard_log_path, histogram_freq=1, embeddings_freq=1),
                              EarlyStopping(monitor='loss', patience=5),
                              EarlyStopping(monitor='val_loss', patience=5),
                              ModelCheckpoint(filepath=cfg.checkpoint_path, monitor='loss', save_best_only=True, save_weights_only=True),
                              ReduceLROnPlateau(monitor='loss', factor=0.05, patience=5)
                              ]
)

location_model.save_weights(cfg.saved_model_weights_path)
