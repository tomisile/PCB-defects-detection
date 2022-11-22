import keras
from keras.layers import BatchNormalization, Activation, Conv2D, Dropout
from keras.layers import Concatenate, AveragePooling2D, Input, MaxPooling2D
from keras.layers import ZeroPadding2D, Flatten, Dense
from keras.models import Model


class DenseNet():
    self.nb_dense_blocks = nb_dense_blocks
    self.channels = channels
    self.model = self.dense_net(x)
    
    def conv_layer(x, channels, dropout_rate=None):

        # BN-ReLU-Conv (1 × 1) (bottleneck layer)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
                filters=4*channels,
                kernel_size=[1,1],
                strides=1,
                activation=None,
                use_bias=False
            )
        # add dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        # BN-ReLU-Conv (3 × 3)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
                filters=channels,
                kernel_size=[3,3],
                padding='same',
                activation=None,
                use_bias=False
            )
        # add dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        
        return x

    def dense_block(x, channels, nb_layers, dropout_rate=None):
        layers = [x]
        for i in range(nb_layers):
            conv_layers = conv_layer(x, channels, dropout_rate)
            layers.append(conv_layers)
            x = Concatenate(axis=-1)(layers)

        return x

    def transition_layer(x, channels, dropout_rate=None):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(
                filters=channels,
                kernel_size=[1,1],
                activation=None,
                use_bias=False
            )
        # add dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        
        x = AveragePooling2D(pool_size=2)(x)

        return x

    def dense_net(channels, input_shape=None, dropout_rate=None,
                nb_dense_blocks=None, nb_classes=None):
        
        img = Input(shape=input_shape)

        x = Conv2D(
                filters=channels,
                kernel_size=[7,7],
                strides=2,
                use_bias=False,
            )
        x = ZeroPadding2D(padding=3)(img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=1)(x)
        x = MaxPooling2D(pool_size=(3,3), strides=2)(x)
        x = dense_block(x, channels, dropout_rate, nb_layers=6)
        x = transition_layer(x, channels, dropout_rate)
        x = dense_block(x, channels, dropout_rate, nb_layers=6)
        x = AveragePooling2D(pool_size=2)(x)
        x = Flatten(data_format='to-search')(x)
        x = Dense(nb_classes, activation='softmax', use_bias=False)(x)

        model_name = 'DenseNet'

        return Model(img, x), model_name

# Todo: create densenet object
#model = DenseNet(input_shape=(64,64,1), nb_dense_block=2)