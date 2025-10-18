"""ReSUnet model implementation (TensorFlow / Keras)"""
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters, kernel_size=3, activation='relu'):
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def residual_block(x, filters):
    shortcut = x
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    x = layers.add([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def encoder_block(x, filters):
    x = residual_block(x, filters)
    p = layers.MaxPool2D((2,2))(x)
    return x, p

def decoder_block(x, skip, filters):
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Concatenate()([x, skip])
    x = residual_block(x, filters)
    return x

def build_resunet(input_shape=(256,256,1), num_classes=1):
    inputs = layers.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    b = residual_block(p3, 256)
    d3 = decoder_block(b, s3, 128)
    d2 = decoder_block(d3, s2, 64)
    d1 = decoder_block(d2, s1, 32)
    outputs = layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d1)
    model = Model(inputs, outputs, name='ReSUnet')
    return model
