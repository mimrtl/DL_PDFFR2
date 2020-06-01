'''
Helena Van Hemmen, May 2020
for PDFF and R2 quantification project
'''

import os
import glob
import nibabel
import numpy as np
import Unet
import tensorflow as tf
from tf.keras.optimizers import Adam
from tf.keras import backend as K

in_rows = 256
in_col = 256


in_ch = 2
out_ch = 3
weightfile_name = 'weights.h5'
data_folder = '/data/data_mrcv2/MCMILLAN_GROUP/50_users/Helena'

def get_myUNet(img_rows,img_cols):
    model = Unet.UNetContinuous([in_rows,in_col,in_ch],out_ch=out_ch,start_ch=16,depth=3,inc_rate=2.,activation='relu',dropout=0.5,batchnorm=True,maxpool=True,upconv=True,residual=False)
    model.compile(optimizer=Adam(lr=1e-4), loss=mean_squared_error, metrics=[mean_squared_error,mean_absolute_error])
    return model

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)
    
def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)

def eval():
        
        np.random.seed(813)
        
        print('Creating/compiling network...')
        model = get_myUNet(in_rows,in_col)
    
        print('Loading saved weights...')
        model.load_weights(weightfile_name)
    
        print('Predicting masks on test data...')
        print('Loading images from ' + data_folder + '...')


