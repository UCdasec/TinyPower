import os
import sys
import argparse
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model


def generate_default_params():
    return {
            'depth': 3,
            'optimizer': "RMSProp",
            'learning_rate': 0.00001,
            'dropout_rate': 0.5,
            'decay': 0.1,
            'ae_batch_size': 100,
            'ae_epochs': 200,
            'batch_size': 100,
            'epochs': 200,
            'data_dim': 350,
            'encoded1_dim': 4096,
            'encoded2_dim': 4096,
            'encoded3_dim': 2048,
            'encoded4_dim': 2048,
            'encoded5_dim': 1024,
            'y_dim': 512,
            'denseLayer': 160,
            'encoded_act': "selu",
            'dense_act': "selu",
            'y_act': "selu",
            'z_act': "selu"
            }


def encoder(inp, params):
    ''' encoder part of autoencoder '''
    print('Setup encoder layers...')
    depth = params['depth']
    for i in range(depth):
        x = Dense(params['encoded{}_dim'.format(i+1)], activation=params['act'], kernel_initializer='glorot_normal')(inp)
        x = BatchNormalization()(x)
        x = Dropout(rate=params['dropout_rate'])(x)

    x = Dense(params['y_dim'], activation=params['y_act'], name='LR', kernel_initializer='glorot_normal')(x)
    mid = BatchNormalization()(x)
    return mid


def decoder(mid, inputDim, params):
    ''' decoder part of autoencoder '''
    print('Setup decoder layers...')
    depth = params['depth']
    # here range should be in a descending order
    for i in range(depth-1, -1, -1):
        x = Dropout(rate=params['dropout_rate'])(mid)
        x = Dense(params['encoded4_dim'.format()], activation=params['encoded_act4'], kernel_initializer='glorot_normal')(x)
        x = BatchNormalization()(x)

    x = Dense(inputDim, activation=params['z_act'], kernel_initializer='glorot_normal')(x)
    out = BatchNormalization()(x)
    return out


def create_autoencoder(inp_shape, params):
    print('create ae model...')
    inp = Input(shape=inp_shape)
    autoencoder = Model(inp, decoder(encoder(inp, params), params))
    autoencoder.compile(optimizer=params['optimizer'], loss='mse')  # reporting the loss
    return autoencoder


def fc(enco, params):
    out_dim = 256
    x = Dense(params['denseLayer'], activation=params['dense_act'], kernel_initializer='glorot_normal')(enco)
    x = BatchNormalization()(x)
    clf = Dense(out_dim, activation='softmax', kernel_initializer='glorot_normal')(x)
    return clf


def create_model(inp_shape, params):
    inputLayer = Input(inp_shape)
    full_model = Model(inputLayer, fc(encoder(inputLayer, params)))
    full_model.compile(loss='categorical_crossentropy',
                       optimizer=params['optimizer'],
                       metrics=['accuracy'])
    return full_model


def plotTheModel():
    inp_shape = (9000, )
    params = generate_default_params()
    autoencoder = create_autoencoder(inp_shape, params)
    full_model = create_model(inp_shape, params)
    picDir = os.path.join('tmp', 'pic')
    if not os.path.isdir(picDir):
        os.makedirs(picDir)
    plot_model(autoencoder, to_file=os.path.join(picDir, 'autoencoder.png'), show_shapes='True')
    plot_model(full_model, to_file=os.path.join(picDir, 'ae_clf.png'), show_shapes='True')


def train_ae(inp_shape, params, X_train, model_path):
    autoencoder = create_autoencoder(inp_shape, params)
    print('Build autoencoder...')

    def lr_scheduler(epoch):
        if epoch % 20 == 0 and epoch != 0:
            lr = K.get_value(autoencoder.optimizer.lr)
            K.set_value(autoencoder.optimizer.lr, lr*params['decay'])
            print("lr changed to {}".format(lr*params['decay']))
        return K.get_value(autoencoder.optimizer.lr)

    ae_modelPath = os.path.join(model_path, 'best_ae_model.h5')
    checkpointer = ModelCheckpoint(filepath=ae_modelPath, monitor='val_loss', verbose=verbose, save_best_only=True, mode='min')
    earlyStopper = EarlyStopping(monitor='val_loss', mode='min', patience=6)
    #scheduler = LearningRateScheduler(lr_scheduler)
    callBack_list = [checkpointer, earlyStopper]

    hist = autoencoder.fit(X_train, X_train, epochs=params['epochs'],
                           batch_size=params['batch_size'],
                           validation_split=0.1, shuffle=True,
                           callbacks=callBack_list)
    return autoencoder


def train_cls(X_train, y_train, params, modelDir, verbose):
    autoencoder = create_autoencoder(inp_shape, params)
    full_model = create_model(inp_shape, params)

    LayNum = len(full_model.layers) - 3
    for l1, l2 in zip(full_model.layers[:LayNum], autoencoder.layers[:LayNum]):
        l1.set_weights(l2.get_weights())

    #for layer in full_model.layers[:LayNum]:
    #    layer.trainable = False

    def lr_scheduler(epoch):
        if epoch % 20 == 0 and epoch != 0:
            lr = K.get_value(full_model.optimizer.lr)
            K.set_value(full_model.optimizer.lr, lr*params['decay'])
            print("lr changed to {}".format(lr*params['decay']))
        return K.get_value(full_model.optimizer.lr)

    #scheduler = LearningRateScheduler(lr_scheduler)
    clf_modelPath = os.path.join(modelDir, 'sae_weights_best.hdf5')
    checkpointer = ModelCheckpoint(filepath=clf_modelPath, monitor='val_acc', verbose=verbose, save_best_only=True, mode='max')
    earlyStopper = EarlyStopping(monitor='val_acc', mode='max', patience=6)
    callBack_list = [checkpointer, earlyStopper]

    print('start to train classifier...')
    hist = full_model.fit(X_train, y_train, batch_size=params['batch_size'],
                          epochs=params['epochs'], verbose=verbose, shuffle=True,
                          validation_split=0.2, callbacks=callBack_list)

    return full_model


def test(X_test, y_test, clf_modelPath):
    print('Predicting results with best model...')
    model = load_model(clf_modelPath)
    score, acc = model.evaluate(X_test, y_test, batch_size=100)
    print('Test score: ', score, '\nTest accuracy: ', acc)
    return acc


def main(opts):
    ''' standalone test '''
    PARAMS = generate_default_params()
    if opts.plotModel:
        plotTheModel()
    else:
        # toy run of the model, to verify it works
        X_train, y_train, X_test, y_test, labelMap, NUM_CLASS = loadData(opts, PARAMS)
        modelPath = train(X_train, y_train, NUM_CLASS)
        acc = sae.test(X_test, y_test, NUM_CLASS, modelPath)
        tmp = '[LOG] -- cnn with data {} accuracy and dataType {} has an accuracy: {:f}'.format(opts.input, opts.dataType, acc)
        print(tmp)
    print('[LOG] -- all done!')


def parseOpts(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help ='file path of config file')
    parser.add_argument('-d', '--dataType',
                        help ='choose from onlyOrder/both')
    parser.add_argument('-o', '--output', default='',
                        help ='output file name')
    parser.add_argument('-m', '--mode', default='',
                        help ='output file name')
    parser.add_argument('-t', '--testOnly', default='',
                        help ='verbose or not')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help ='verbose or not')
    parser.add_argument('-p', '--plotModel', action='store_true',
                        help ='verbose or not')
    opts = parser.parse_args()
    return opts


if __name__=="__main__":
    opts = parseOpts(sys.argv)
    if tf.test.is_gpu_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main(opts)

