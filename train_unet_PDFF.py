'''
Helena Van Hemmen, May 2020
for PDFF and R2 quantification project
'''
import os
import glob
import numpy as np
import tensorflow
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import mean_absolute_error, mean_squared_error
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback, History
from time import time
from matplotlib import pyplot as plt
import unet

in_ch = 2
out_ch = 3

in_rows = 256
in_col = 256
data_folder = '/data/data_mrcv2/MCMILLAN_GROUP/50_users/Helena/Ideal_data1p5'

batch_size = 15
epochs = 100

def get_unet(in_rows,in_col,in_ch):
    model = unet.unet((in_rows,in_col,in_ch),out_ch=out_ch,start_ch=16,depth=3,inc_rate=2.,activation='relu',dropout=.5,batchnorm=False,maxpool=True,upconv=True,residual=False)
    model.compile(optimizer=Adam(lr=1e-4),loss=mean_squared_error,metrics=[mean_squared_error,mean_absolute_error])
    model.summary()
    return model

def train():
    
    weight_filename = 'weights.h5'

    print('Creating/compiling network...')

    model = get_unet(in_rows,in_col,in_ch)
    model_checkpoint = ModelCheckpoint(weight_filename,monitor='loss',save_best_only=True)

    print('Loading data...')

    in_echo1 = np.load('input_echo1.npy')
    out = np.load('output.npy')

    datagen1 = ImageDataGenerator(
        rotation_range=15,
        shear_range=10,
        width_shift_range=0.33,
        height_shift_range=0.33,
        zoom_range=0.33,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2)
    datagen2 = ImageDataGenerator(
        rotation_range=15,
        shear_range=10,
        width_shift_range=0.33,
        height_shift_range=0.33,
        zoom_range=0.33,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2)

    seed = 1
    datagen1.fit(in_echo1,seed=seed)
    datagen2.fit(out,seed=seed)

    batchsize = 20
    n_epochs = 200

    datagen = zip( datagen1.flow( in_echo1, None, batchsize, seed=seed, subset="training"), datagen2.flow( out, None, batchsize, seed=seed, subset="training") )
    datagen_val = zip( datagen1.flow( in_echo1, None, batchsize, seed=seed, subset="validation"), datagen2.flow( out, None, batchsize, seed=seed, subset="validation") )
    
    print('Fitting network...')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    history = History()
    fig = plt.figure(figsize=(15,5))
    fig.show(False)
    #display_progress = LambdaCallback(on_epoch_end = lambda epoch, logs: progresscallback_img2img(epoch,logs,model,history,fig,in_echo1[150,:,:,:],out[150,:,:,:]))
    model.fit_generator( datagen, steps_per_epoch=500, validation_data=datagen_val, validation_steps=250, epochs=n_epochs, callbacks=[model_checkpoint,tensorboard,history] )
    

def progresscallback_img2img(epoch,logs,model,history,fig,input_x,target_y):
    fig.clf()
    a = fig.add_subplot(1,4,1)
    plt.imshow(np.rot90(input_x[:,:,0]),cmap='gray')  
    a.axis('off')
    a.set_title('input')
    a = fig.add_subplot(1,4,2)
    plt.imshow(np.rot90(target_y[:,:,0]),cmap='gray')
    a.axis('off')
    a.set_title('target')
    a = fig.add_subplot(1,4,3)
    pred_y = model.predict(np.expand_dims(input_x,axis=0))
    plt.imshow(np.rot90(np.squeeze(pred_y[0,:,:,0])),cmap='gray')
    a.axis('off')
    a.set_title('prediction at epoch'+repr(epoch+1))
    a = fig.add_subplot(1,4,4)
    plt.plot(range(epoch+1),history.history['loss'],'b',label = 'training loss')
    plt.plot(range(epoch+1),history.history['val_loss'],'r',label='validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    a.set_title('Losses')
    fig.tight_layout()
    fig.canvas.draw()
    fig.savefig('progress_image_fixnametodo_{0:05d}.jpg'.format(epoch+1))
    fig.canvas.flush_events()

if __name__ == '__main__':
    train()
