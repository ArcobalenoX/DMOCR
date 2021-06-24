from tensorflow.keras import backend as ktf
from tensorflow.keras import layers
from tensorflow.keras.layers import (GRU, LSTM, Activation, Add, Attention,
                                     BatchNormalization, Bidirectional,
                                     Concatenate, Conv1D, Conv2D, Dense,
                                     Dropout, Flatten, Input, Lambda, Layer,
                                     MaxPooling2D, Permute, Reshape,
                                     SeparableConv2D, TimeDistributed)
from tensorflow.keras.models import Model


class separableconv(Layer):
    """
    @Separable Convolutional Block Module
    """

    def __init__(self, filters, strides=(1, 1), **kwargs):
        super(separableconv, self).__init__(**kwargs)

        self.sconv1 = SeparableConv2D(filters, (3, 3), padding='same')
        self.sbn1 = BatchNormalization()
        self.sact1 = Activation('swish')
        self.sconv2 = SeparableConv2D(
            filters, (3, 3), strides=strides, padding='same')
        self.sbn2 = BatchNormalization()
        self.sact2 = Activation('swish')

    def call(self, inputs, **kwargs):

        y = self.sconv1(inputs)
        y = self.sbn1(y)
        y = self.sact1(y)
        y = self.sconv2(y)
        y = self.sbn2(y)
        y = self.sact2(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        return config


class normalconv(Layer):
    """
    @Normal Convolutional Block Module
    """

    def __init__(self, filters, strides=(1, 1), **kwargs):
        super(normalconv, self).__init__(**kwargs)

        self.conv1 = Conv2D(filters, (3, 3), padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('swish')
        self.conv2 = Conv2D(filters, (3, 3), padding='same')
        self.bn2 = BatchNormalization()
        self.act2 = Activation('swish')
        self.pool = MaxPooling2D((2, 2), strides=strides, padding='same')

    def call(self, inputs, **kwargs):

        y = self.conv1(inputs)
        y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act2(y)
        y = self.pool(y)

        return y

    def get_config(self):
        config = super().get_config().copy()
        return config


class CRNN:
    def __init__(self, height, width,  label_len, characters):
        self.height = height
        self.width = width
        self.label_len = label_len
        self.characters = characters
        self.label_classes = len(self.characters)

    def ctc_loss(self, args):
        iy_pred, ilabels, iinput_length, ilabel_length = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        iy_pred = iy_pred[:, 2:, :]  # no such influence
        return ktf.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)

    def network(self):

        input_img = Input(shape=(self.height, self.width, 3))
        x = separableconv(64)(input_img)
        x = separableconv(64)(x)
        x = separableconv(128)(x)
        x = separableconv(128, strides=(2, 2))(x)
        x = separableconv(256)(x)
        x = separableconv(256, strides=(2, 2))(x)
        x = separableconv(512)(x)
        x = separableconv(512, strides=(2, 2))(x)

        x = Permute((2, 1, 3))(x)
        x = TimeDistributed(Flatten(), name='timedistrib')(x)

        fc1 = Dense(256, activation='relu')(x)
        bi_rnn1 = Bidirectional(GRU(128, return_sequences=True), merge_mode='sum')(fc1)
        bi_rnn2 = Bidirectional(GRU(128, return_sequences=True), merge_mode='concat')(bi_rnn1)
        #x = Dropout(0.25)(bi_rnn2)
        fc2 = Dense(self.label_classes, activation='softmax')(bi_rnn2)

        infer_model = Model(inputs=input_img, outputs=fc2)

        labels = Input(name='the_labels', shape=[self.label_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(self.ctc_loss, output_shape=(1,), name='ctc')([fc2, labels, input_length, label_length])
        train_model = Model(inputs=[input_img, labels, input_length, label_length], outputs=[loss_out])

        return train_model, infer_model


if __name__ == '__main__':
    train_model, infer_model = CRNN(64, 128, 11, '0123456789.-|').network()
    train_model.summary()
    # infer_model.summary()
