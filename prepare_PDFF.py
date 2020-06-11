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
data_folder = '/data/data_mrcv2/MCMILLAN_GROUP/50_users/Helena/Ideal_data1p5'


def prepare():
    
    np.random.seed(813)

    print('Loading images from ' + data_folder + '...')

    input_count = 0
    for filename in os.listdir(data_folder):
        if (not filename.startswith('00')) and filename.endswith('.mat'):
        
            input_count += 1

            print(filename)
            curr_input = spio.loadmat(os.path.join(data_folder,filename), struct_as_record = True)

        
            img_idata = curr_input['img_idata']
            print('img_idata: ')
            print(img_idata.shape)
            img_ref = curr_input['img_ref']
            print('img_ref: ')
            print(img_ref.shape)

            in_slices = img_idata.shape[2]

            input_echo1 = np.zeros([in_rows,in_col,in_slices,in_ch])
            output = np.zeros([in_rows,in_col,in_slices,out_ch])

            curr_in = img_idata[:,:,:,0:2]
            print('curr_in: ')
            print(curr_in.shape)
            curr_out = img_ref[:,:,:,:]
            print('curr_out: ')
            print(curr_out.shape)

            if input_count == 1:
                input_echo1 = curr_in
                output = curr_out
            elif input_count != 1:
                input_echo1 = np.concatenate((input_echo1,curr_in),axis=2)
                output = np.concatenate((output,curr_out),axis=2)

            print('count = {}'.format(input_count))
            
            if input_count == 120:
                break
            
            input_echo1 = np.swapaxes(input_echo1,2,0)
            output = np.swapaxes(output,2,0)
            print(input_echo1.shape)
            print(output.shape)

            np.save('input_echo1.npy',input_echo1)
            np.save('output.npy',output)

        
        else:
            continue

prepare()


        
