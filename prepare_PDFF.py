'''
Helena Van Hemmen, May 2020
for PDFF and R2 quantification project
'''
import os
import glob
import numpy as np
import scipy.io as spio

in_ch = 2
out_ch = 3

in_rows = 256
in_col = 256
data_folder = '/data/data_mrcv2/MCMILLAN_GROUP/50_users/Helena'


def prepare():
    
    #np.random.seed(813)

    print('Loading images from' + data_folder + '...')

    input_count = 0
    for filename in os.listdir(data_folder):
        if (not filename.startswith('00')) and filename.endswith('.mat'):
        
            input_count += 1

            curr_input = spio.loadmat(filename, struct_as_record = True)

        
            img_idata = curr_input['img_idata']
            img_ref = curr_input['img_ref']

            in_slices = img_idata.shape[2]

            input_echo1 = np.zeros([in_rows,in_col,in_slices,in_ch])
            output = np.zeros([in_rows,in_col,in_slices,out_ch])

            if input_count == 1:
                input_echo1 = img_idata[:,:,:,0:1]
                output = img_ref[:,:,:,:]
            else:
                input_echo1 = np.concatenate((input_echo1,img_idata[:,:,:,0:1]),axis=0)
                output = np.concatenate((output,img_ref[:,:,:,0:2]),axis=0)

            print('count = {}'.format(input_count))

            if input_count == 120:
                break

            np.save('input_echo1.npy',input_echo1)
            np.save('output.npy',output)

        
        else:
            continue






        