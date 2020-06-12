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
            img_ref = curr_input['img_ref']

            in_slices = img_idata.shape[2]

            input_echo1 = np.zeros([in_slices,in_rows,in_col,in_ch])
            output = np.zeros([in_slices,in_rows,in_col,out_ch])

            curr_in = img_idata[:,:,:,0:2]
            print('curr_in before swap: ')
            print(curr_in.shape)
            curr_out = img_ref[:,:,:,:]
            print('curr_out before swap: ')
            print(curr_out.shape)

            curr_in = np.transpose(curr_in,(2,0,1,3))
            curr_out = np.transpose(curr_out,(2,0,1,3))
            print('curr_in after swap: ')
            print(curr_in.shape)
            print('curr_out after swap: ')
            print(curr_out.shape)


            if input_count == 1:
                input_echo1 = curr_in
                print('input_echo1 after count=1: ')
                print(input_echo1.shape)
                output = curr_out
                print('output after count=1: ')
                print(output.shape)
            else:
                print('index 1 shape: ')
                print(curr_in.shape)
                input_echo1 = np.concatenate((input_echo1,curr_in))
                print(curr_out.shape)
                output = np.concatenate((output,curr_out))

            print('count = {}'.format(input_count))
            print('final shape in: ')
            print(input_echo1.shape)
            print('final shape out: ')
            print(output.shape)
            
            if input_count == 60:
                break

            np.save('input_echo1.npy',input_echo1)
            np.save('output.npy',output)

        
        else:
            continue

prepare()


        
