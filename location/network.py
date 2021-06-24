# coding=utf-8
import math
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Input, Dense, BatchNormalization, Activation, Concatenate, Add,Dropout
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import plot_model
import cfg

class Att(Layer):
    """
    @simple Attention Module
    """
    def __init__(self, **kwargs):
        super(Att, self).__init__(**kwargs)

    def call(self, layer_input, **kwargs):
        x = layer_input
        attention = tf.reduce_mean(layer_input, axis=-1, keepdims=True) # (B, W, H, 1)
        importance_map = tf.sigmoid(attention)
        output = tf.multiply(x, importance_map) # (B, W, H, C)
        return output
    def compute_output_shape(self, input_shape):
        return input_shape



class CBAM(Layer):
    """
    @Convolutional Block Attention Module
    """

    def __init__(self,**kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.conv=Conv2D(1,7, padding='same', activation='sigmoid')

    def call(self, layer_input, **kwargs):
        x = layer_input
        b,w,h,c=layer_input.shape

        # channel attention
        x_mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        x_max = tf.reduce_max(x, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
        x = tf.add(x_mean, x_max)  # (B, 1, 1, C)
        x = tf.nn.sigmoid(x)  # (B, 1, 1, C)
        x = tf.multiply(layer_input, x)  # (B, W, H, C)

        # spatial attention
        y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
        y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
        y = tf.concat([y_mean, y_max], axis=-1)  # (B, W, H, 2)
        y = self.conv(y)  # (B, W, H, 1)

        # series
        z = tf.multiply(x, y)  # (B, W, H, C)
        return z

class separableconv(Layer):
    """
    @Separable Convolutional Block Module
    """

    def __init__(self, filters, strides=(1,1), **kwargs):
        super(separableconv, self).__init__(**kwargs)
        
        self.sconv1 = SeparableConv2D(filters, (3, 3), padding='same')
        self.sbn1 = BatchNormalization()
        self.sact1 = Activation('relu')
        self.sconv2 = SeparableConv2D(filters, (3, 3), strides=strides,padding='same')
        self.sbn2  = BatchNormalization()
        self.sact2 = Activation('relu')

    def call(self, inputs, **kwargs):

        y=self.sconv1(inputs)
        y=self.sbn1(y)
        y=self.sact1(y)
        y=self.sconv2(y)
        y=self.sbn2(y)
        y=self.sact2(y)
        return y

    def get_config(self):
        config = super().get_config().copy()
        return config

class normalconv(Layer):
    """
    @Normal Convolutional Block Module
    """
    def __init__(self,filters,strides=(1,1),**kwargs):
        super(normalconv, self).__init__(**kwargs)
        
        self.conv = Conv2D(filters, (3, 3), strides=strides, padding='same')
        self.bn = BatchNormalization()
        self.act = Activation('relu')

    def call(self, inputs, **kwargs):

        y=self.conv(inputs)
        y=self.bn(y)
        y=self.act(y)
        return y

    def get_config(self):
        config = super().get_config().copy()
        return config


class Location:

    def __init__(self):
        self.input_img = Input(shape=(cfg.image_size, cfg.image_size, 3),
                               name='input_img', 
                               dtype='float32')

    def location_network(self):

        # print('input_shape',self.input_img.shape)
        #block0
        x = self.input_img
        x = separableconv(32, name='conv0_0')(x)
        x = separableconv(32, strides=(2,2), name='conv0_1')(x)
        x = MaxPooling2D((2, 2),padding='same')(x)
        layer0 = x

        ##block1
        residual = separableconv(64, name='shortcut1_1')(x)
        residual = separableconv(64, strides=(2,2), name='shortcut1_2')(residual)
        x = separableconv(64, name='conv1')(x)
        x = separableconv(64, strides=(2,2), name='scon1')(x)
        x = Add()([x, residual])
        layer1 = x

        ##block2
        residual = separableconv(128, name='shortcut2_0')(x)
        residual = separableconv(128, strides=(2,2), name='shortcut2_1')(residual)
        x = separableconv(128, name='conv2')(x)
        x = separableconv(128, strides=(2,2), name='scon2')(x)
        x = Add()([x, residual])
        layer2 = x

        #bolck3
        residual = separableconv(256, name='shortcut3_0')(x)
        residual = separableconv(256, strides=(2,2), name='shortcut3_1')(residual)
        x = separableconv(256, name='conv3')(x)
        x = separableconv(256, strides=(2,2), name='scon3')(x)
        x = Add()([x, residual])
        layer3 = x

        uplayer3=Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(layer3)
        d1 = Concatenate(axis=-1, name='merge1')([uplayer3, layer2])
        d1 = BatchNormalization()(d1)
        d1 = separableconv(128)(d1)
        d1 = CBAM(name='attention1')(d1)
        d1 = Dropout(0.5)(d1)

        uplayer2 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(d1)
        d2 = Concatenate(axis=-1, name='merge2')([uplayer2, layer1])
        d2 = BatchNormalization()(d2)
        d2 = separableconv(64)(d2)
        d2 = CBAM(name='attention2')(d2)
        d2 = Dropout(0.5)(d2)
        
        uplayer1 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(d2)
        d3 = Concatenate(axis=-1, name='merge3')([uplayer1, layer0])
        d3 = BatchNormalization()(d3)
        d3 = separableconv(32)(d3)
        d3 = CBAM(name='attention3')(d3)
        d3 = Dropout(0.5)(d3)
        
        
        before_output = Conv2D(32,(3,3),padding='same')(d3)

        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(before_output) #score map
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_output) #coordinate (x,y)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_output)
        detection = Concatenate(axis=-1, name='detection')([inside_score, side_v_code, side_v_coord])

        model = Model(inputs=self.input_img, outputs=[detection], name='LocationModel')
        return model



#efficient

def swish(x):
    return x * tf.nn.sigmoid(x)

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = None
    min_depth = min_depth or depth_divisor
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, input_channels, ratio=0.25):
        super(SEBlock, self).__init__()
        self.num_reduced_filters = max(1, int(input_channels * ratio))
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")
        self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                  kernel_size=(1, 1),
                                                  strides=1,
                                                  padding="same")

    def call(self, inputs, **kwargs):
        branch = self.pool(inputs)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = tf.expand_dims(input=branch, axis=1)
        branch = self.reduce_conv(branch)
        branch = swish(branch)
        branch = self.expand_conv(branch)
        branch = tf.nn.sigmoid(branch)
        output = inputs * branch
        return output
    def get_config(self):
        config = super().get_config().copy()
        return config

class MBConv(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
        super(MBConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                      strides=stride,
                                                      padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(input_channels=in_channels * expansion_factor)
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                            kernel_size=(1, 1),
                                            strides=1,
                                            padding="same")
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = swish(x)
        x = self.dwconv(x)
        x = self.bn2(x, training=training)
        x = self.se(x)
        x = swish(x)
        x = self.conv2(x)
        x = self.bn3(x, training=training)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_rate:
                x = self.dropout(x, training=training)
            x = tf.keras.layers.add([x, inputs])
        return x
    def get_config(self):
        config = super().get_config().copy()
        return config

class Location_efficient:
    def __init__(self):
        self.input_img = Input(shape=(cfg.image_size, cfg.image_size, 3),
                               name='input_img', 
                               dtype='float32')

    def location_network(self):

        # print('input_shape',self.input_img.shape)
        #block0
        x = self.input_img

        x = MBConv(in_channels=3,
                             out_channels=32,
                             expansion_factor=1,
                             stride=2,
                             k=3,
                             drop_connect_rate=0.1)(x)
        layer0 = x

        ##block1
        residual = separableconv(64, name='shortcut1_1')(x)
        residual = separableconv(64, strides=(2,2), name='shortcut1_2')(residual)
        x = MBConv(in_channels=32,
                             out_channels=64,
                             expansion_factor=1,
                             stride=2,
                             k=3,
                             drop_connect_rate=0.2)(x)
        x = Add()([x, residual])
        layer1 = x

        ##block2
        residual = separableconv(128, name='shortcut2_0')(x)
        residual = separableconv(128, strides=(2,2), name='shortcut2_1')(residual)
        x = MBConv(in_channels=64,
                             out_channels=128,
                             expansion_factor=1,
                             stride=2,
                             k=3,
                             drop_connect_rate=0.3)(x)
        x = Add()([x, residual])
        layer2 = x

        #bolck3
        residual = separableconv(256, name='shortcut3_0')(x)
        residual = separableconv(256, strides=(2,2), name='shortcut3_1')(residual)
        x = MBConv(in_channels=128,
                             out_channels=256,
                             expansion_factor=1,
                             stride=2,
                             k=3,
                             drop_connect_rate=0.4)(x)
        x = Add()([x, residual])
        layer3 = x

        uplayer3=Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(layer3)
        d1 = Concatenate(axis=-1, name='merge1')([uplayer3, layer2])
        d1 = BatchNormalization()(d1)
        d1 = separableconv(128)(d1)
        d1 = CBAM(name='attention1')(d1)
        d1 = Dropout(0.5)(d1)

        uplayer2 = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same')(d1)
        d2 = Concatenate(axis=-1, name='merge2')([uplayer2, layer1])
        d2 = BatchNormalization()(d2)
        d2 = separableconv(64)(d2)
        d2 = CBAM(name='attention2')(d2)
        d2 = Dropout(0.5)(d2)
        
        uplayer1 = Conv2DTranspose(32, (3,3), strides=(2,2), padding='same')(d2)
        d3 = Concatenate(axis=-1, name='merge3')([uplayer1, layer0])
        d3 = BatchNormalization()(d3)
        d3 = separableconv(32)(d3)
        d3 = CBAM(name='attention3')(d3)
        d3 = Dropout(0.5)(d3)
        
        before_output = separableconv(32, strides=(2,2))(d3)

        inside_score = Conv2D(1, 1, padding='same', name='inside_score')(before_output)
        side_v_code = Conv2D(2, 1, padding='same', name='side_vertex_code')(before_output)
        side_v_coord = Conv2D(4, 1, padding='same', name='side_vertex_coord')(before_output)
        location = Concatenate(axis=-1, name='detection')([inside_score, side_v_code, side_v_coord])

        model = Model(inputs=self.input_img, outputs=[location], name='LocationModel')
        return model





if __name__ == "__main__":
    #location_model = Location().location_network()
    location_model = Location_efficient().location_network()
    location_model.summary()

    #plot_model(location_model,'location.jpg')
