'''
Helena Van Hemmen, May 2020
for PDFF and R2 quantification project
'''
import os
import glob
import numpy as np
import scipy.io as spio
import pandas as pd

in_ch = 12
out_ch = 3

in_rows = 256
in_col = 256
data_folder = '/data/data_mrcv2/MCMILLAN_GROUP/50_users/Helena'


def prepare():
    
    np.random.seed(813)

    print('Loading images from' + data_folder + '...')

    input_count = 0
    for filename in os.listdir(data_folder):
        if (not filename.startswith('00')) and filename.endswith('.mat'):
        
            input_count += 1

            curr_input = spio.loadmat(filename, struct_as_record = True)

        
            img_idata = curr_input['img_idata']
            img_ref = curr_input['img_ref']

            in_slices = img_idata.shape[2]

            input1_echo1 = np.zeros([in_rows,in_col,in_slices,1])
            input2_echo1 = np.zeros([in_rows,in_col,in_slices,1])
            output_fat = np.zeros([in_rows,in_col,in_slices,1])
            output_water = np.zeros([in_rows,in_col,in_slices,1])
            output_r2 = np.zeros([in_rows,in_col,in_slices,1])

            if input_count == 1
                input1_echo1 = img_idata[:,:,:,1]
                input2_echo1 = img_idata[:,:,:,2]
                output_water = img_ref[:,:,:,1]
                output_fat = img_ref[:,:,:,2]
                output_r2 = img_ref[:,:,:,3]
            else:
                input1_echo1 = np.concatenate((input1_echo1,img_idata[:,:,:,1]),axis=0)
                input2_echo1 = np.concatenate((input1_echo2,img_idata[:,:,:,2]),axis=0)
                output_water = np.concatenate((output_water,img_ref[:,:,:,1]),axis=0)
                output_fat = np.concatenate((output_fat,img_ref[:,:,:,2]),axis=0)
                output_r2 = np.concatenate((output_r2,img_ref[:,:,:,3]),axis=0)

            print('count = {}'.format(img_count))

            if input_count = 120
                break

            np.save('input1_echo1.npy',input1_echo1)
            np.save('input2_echo1.npy',input2_echo1)
            np.save('output_water.npy',output_water)
            np.save('output_fat.npy',output_fat)
            np.save('output_r2.npy',output_r2)

        
        else:
            continue






        