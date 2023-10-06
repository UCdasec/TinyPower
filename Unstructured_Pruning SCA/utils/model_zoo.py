#!/usr/bin/python3.6
import pdb
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, AveragePooling1D ,Flatten, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import RMSprop, Adam
import numpy as np
import pandas as pd

def cnn_rs_xmega(input_shape, r = [1] * 7, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(2*r[0]), 75, strides=1, activation='selu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(75, strides=50, name='pool1')(x)
    x = Conv1D(int(4*r[1]), 2, strides=1, activation='selu', padding='same', name='block1_conv2')(x)
    x = AveragePooling1D(2, strides=2, name='pool2')(x)
    x = Conv1D(int(2*r[2]), 3, strides=1, activation='selu', padding='same', name='block1_conv3')(x)
    x = AveragePooling1D(7, strides=7, name='pool3')(x)
    x = GlobalAveragePooling1D()(x)
    # x = Flatten(name='flatten')(x)
    x = Dense(int(32*r[3]), activation='selu', name='fc1')(x)
    x = Dense(int(2*r[4]), activation='selu', name='fc2')(x)
    # x = Dense(int(9*r[3]), activation='selu', name='fc3')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model
    
def cnn_rs_stm_2(input_shape, r = [1] * 7, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(8*r[0]), 50, strides=1, activation='selu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(100, strides=25, name='pool1')(x)
    x = Conv1D(int(2*r[1]), 2, strides=1, activation='selu', padding='same', name='block1_conv2')(x)
    x = AveragePooling1D(25, strides=25, name='pool2')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(int(16*r[2]), activation='selu', name='fc1')(x)
    x = Dense(int(2*r[3]), activation='selu', name='fc3')(x)
    
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model

def cnn_rs_stm(input_shape, r = [1] * 7, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(4*r[0]), 25, strides=1, activation='selu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(75, strides=50, name='pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(int(2*r[1]), activation='selu', name='fc1')(x)
    
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model

### CNN Best model
def cnn_best(input_shape, r = [1] * 7, emb_size=256, classification=True):

    # r = [1] * 7
   
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(64*r[0]), 11, strides=2, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
    # x = BatchNormalization()(x)
    # Block 2
    x = Conv1D(int(128*r[1]), 11, activation='relu', padding='same', name='block2_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
    # x = BatchNormalization()(x)
    # Block 3
    x = Conv1D(int(256*r[2]), 11, activation='relu', padding='same', name='block3_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
    # x = BatchNormalization()(x)
    # Block 4
    x = Conv1D(int(512*r[3]), 11, activation='relu', padding='same', name='block4_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
    # x = BatchNormalization()(x)
    # Block 5
    x = Conv1D(int(512*r[4]), 11, activation='relu', padding='same', name='block5_conv1')(x)
    x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
    # x = BatchNormalization()(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(int(4096*r[5]), activation='relu', name='fc1')(x)
    x = Dense(int(4096*r[6]), activation='relu', name='fc2')(x)
    if classification:
        x = Dense(emb_size, activation='softmax', name='predictions')(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        optimizer = RMSprop(lr=0.00001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print('[log] --- finish construct the cnn2 model')
        return model
    else:
        # embeddings = x
        x = Dense(emb_size, kernel_initializer="he_normal")(x)
        # Create model.
        model = Model(inp, x, name='cnn_best')
        return model

def copy_modified_weights(weights1,source_model, target_model,debug= False):
    i=0

    if debug == True:
        print(weights1[0])

    for source_layer, target_layer in zip(source_model.layers, target_model.layers):
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()
        if ("conv" in target_layer.name ) and target_layer != target_model.layers[-1]:
            source_weights[1]=weights1[i]
            i=i+1
            target_layer.set_weights(source_weights)
            if debug == True:
                print("conv weights coppied")
                print(np.shape(source_weights[1]))
        elif "Dense" == (source_layer.__class__.__name__ )and target_layer != target_model.layers[-1]:
            source_weights[1] = weights1[i]
            i += 1
            target_layer.set_weights(source_weights)
            
            if debug == True:
                print("Dense weights coppied")
                print(np.shape(source_weights[1]))
        else:
            target_layer.set_weights(source_weights)
    return target_model


if __name__ == '__main__':
    input_shape = (1000, 1)
    r = [1] * 7
    model = cnn_rs_stm(input_shape, r, emb_size=9, classification=True)
    print("!!!")
    model.summary()