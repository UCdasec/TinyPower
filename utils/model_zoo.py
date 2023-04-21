#!/usr/bin/python3.6
import pdb
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Input, AveragePooling1D ,Flatten, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd

def cnn_rs_xmega(input_shape, r = [1] * 7, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(2*r[0]), 75, strides=1, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(75, strides=50, name='pool1')(x)
    x = Conv1D(int(4*r[0]), 2, strides=1, activation='relu', padding='same', name='block1_conv2')(x)
    x = AveragePooling1D(2, strides=2, name='pool2')(x)
    x = Conv1D(int(2*r[0]), 3, strides=1, activation='relu', padding='same', name='block1_conv3')(x)
    x = AveragePooling1D(7, strides=7, name='pool3')(x)
    x = GlobalAveragePooling1D()(x)
    # x = Flatten(name='flatten')(x)
    x = Dense(int(32*r[1]), activation='selu', name='fc1')(x)
    x = Dense(int(4*r[2]), activation='selu', name='fc2')(x)
    # x = Dense(int(9*r[3]), activation='selu', name='fc3')(x)
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
    

def cnn_rs_stm(input_shape, r = [1] * 7, emb_size=256, classification=True):
    inp = Input(shape=input_shape)
    # Block 1
    x = Conv1D(int(4*r[0]), 25, strides=1, activation='relu', padding='same', name='block1_conv1')(inp)
    x = AveragePooling1D(75, strides=50, name='pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(int(2*r[1]), activation='selu', name='fc1')(x)
    
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


def copy_weights(pre_trained_model, target_model, ranks_path):
    ranks = pd.read_csv(ranks_path, header=None).values

    rr = list()
    for r in ranks:
        r = r[~np.isnan(r)]
        r = list(map(int, r))
        rr.append(r)

    for l_idx, l in enumerate(target_model.layers):
        if "conv" in l.name:
            conv_id = int(l.name[5]) - 1
            this_idcies = rr[conv_id][1:l.filters + 1]
            if conv_id == 0 :
                weights = pre_trained_model.layers[l_idx].get_weights()[0][:, :, this_idcies]
            else:
                last_idcies = rr[conv_id - 1][1:last_filters + 1]
                weights = pre_trained_model.layers[l_idx].get_weights()[0][:, :, this_idcies]
                weights = weights[:, last_idcies,:]

            bias = pre_trained_model.layers[l_idx].get_weights()[1][this_idcies]
            l.set_weights([weights, bias])
            last_filters = l.filters
    return target_model


if __name__ == '__main__':
    input_shape = (1000, 1)
    r = [1] * 7
    model = cnn_rs_stm(input_shape, r, emb_size=9, classification=True)
    print("!!!")
    model.summary()