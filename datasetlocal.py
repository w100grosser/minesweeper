from data_gen import gen_input_output
from tqdm import tqdm

import numpy as np

train = 100
test = 50
val = 50

train_path = 'datasetnewlocal/train/'
test_path = 'datasetnewlocal/test/'
val_path = 'datasetnewlocal/val/'

iters = 0
for i in tqdm(range(train), desc = 'Generating train data', unit = 'data', total = train):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    input_data, output_data = input_array, target_array
    input_array = np.zeros((input_data.shape[0], input_data.shape[1] + 6, input_data.shape[2] + 6))
    input_array[0] = -1
    input_array[:,3:-3,3:-3] = input_data
    for i in range(output_data.shape[0]):
        for j in range(output_data.shape[1]):
            target = output_data[i,j]
            local_input_array = input_array[:,i:i+7,j:j+7]
            np.save(train_path + 'input_' + str(iters), local_input_array)
            np.save(train_path + 'target_' + str(iters), target)
            iters += 1
    

iters = 0
for i in tqdm(range(test), desc = 'Generating test data', unit = 'data', total = test):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    input_data, output_data = input_array, target_array
    input_array = np.zeros((input_data.shape[0], input_data.shape[1] + 6, input_data.shape[2] + 6))
    input_array[0] = -1
    input_array[:,3:-3,3:-3] = input_data
    for i in range(output_data.shape[0]):
        for j in range(output_data.shape[1]):
            target = output_data[i,j]
            local_input_array = input_array[:,i:i+7,j:j+7]
            np.save(test_path + 'input_' + str(iters), local_input_array)
            np.save(test_path + 'target_' + str(iters), target)
            iters += 1
    
iters = 0
for i in tqdm(range(val), desc = 'Generating val data', unit = 'data', total = val):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    input_data, output_data = input_array, target_array
    input_array = np.zeros((input_data.shape[0], input_data.shape[1] + 6, input_data.shape[2] + 6))
    input_array[0] = -1
    input_array[:,3:-3,3:-3] = input_data
    for i in range(output_data.shape[0]):
        for j in range(output_data.shape[1]):
            target = output_data[i,j]
            local_input_array = input_array[:,i:i+7,j:j+7]
            np.save(val_path + 'input_' + str(iters), local_input_array)
            np.save(val_path + 'target_' + str(iters), target)
            iters += 1