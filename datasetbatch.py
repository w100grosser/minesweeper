from data_gen import gen_input_output
from tqdm import tqdm

import numpy as np

train = 500
test = 200
val = 200

train_path = 'datasetnewbatch/train/'
test_path = 'datasetnewbatch/test/'
val_path = 'datasetnewbatch/val/'

for i in tqdm(range(train), desc = 'Generating train data', unit = 'data', total = train):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    X_batch = []
    Y_batch = []
    for t in range(8):
        input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
        input_data = input_array.flatten()
        output_data = target_array.flatten()
        X_batch.append(input_data)
        Y_batch.append(output_data)

    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    np.save(train_path + 'input_' + str(i), X_batch)
    np.save(train_path + 'target_' + str(i), Y_batch)
    

for i in tqdm(range(test), desc = 'Generating test data', unit = 'data', total = test):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    X_batch = []
    Y_batch = []
    for t in range(8):
        input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
        input_data = input_array.flatten()
        output_data = target_array.flatten()
        X_batch.append(input_data)
        Y_batch.append(output_data)

    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    np.save(test_path + 'input_' + str(i), X_batch)
    np.save(test_path + 'target_' + str(i), Y_batch)
    
for i in tqdm(range(val), desc = 'Generating val data', unit = 'data', total = val):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    X_batch = []
    Y_batch = []
    for t in range(8):
        input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
        input_data = input_array.flatten()
        output_data = target_array.flatten()
        X_batch.append(input_data)
        Y_batch.append(output_data)

    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    np.save(val_path + 'input_' + str(i), X_batch)
    np.save(val_path + 'target_' + str(i), Y_batch)