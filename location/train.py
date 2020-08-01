import os
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam,Nadam
import tensorflow.keras.backend as K

from data_loader import gen
from losses import quad_loss
from network import Location
import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

location_model = Location().location_network()
location_model.summary()

Nadamopt=Adam(lr=0.001)
location_model.compile(loss=quad_loss,optimizer=Nadamopt)#,metrics=['Recall','Precision'])
#location_model.load_weights("saved_model/location_weights.h5")
location_model.fit(gen(),
                           steps_per_epoch=cfg.steps_per_epoch,
                           initial_epoch=0,
                           epochs=25,
                           validation_data=gen(is_val=True),
                           validation_steps=cfg.validation_steps,
                           callbacks=[TensorBoard('logs'),
                                      EarlyStopping(monitor='loss',patience=5),
                                      EarlyStopping(monitor='val_loss',patience=5),
                                      ModelCheckpoint(filepath=cfg.checkpoint_path,monitor='loss',save_best_only=True,save_weights_only=True)
                                      ]
                           )
                           
location_model.save_weights(cfg.saved_model_weights_path)
