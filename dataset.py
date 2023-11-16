from data_gen import gen_input_output
from tqdm import tqdm

import numpy as np

train = 50000
test = 20000
val = 20000

train_path = 'datasetnew/train/'
test_path = 'datasetnew/test/'
val_path = 'datasetnew/val/'

for i in tqdm(range(train), desc = 'Generating train data', unit = 'data', total = train):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    np.save(train_path + 'input_' + str(i), input_array)
    np.save(train_path + 'target_' + str(i), target_array)
    

for i in tqdm(range(test), desc = 'Generating test data', unit = 'data', total = test):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    np.save(test_path + 'input_' + str(i), input_array)
    np.save(test_path + 'target_' + str(i), target_array)
    
for i in tqdm(range(val), desc = 'Generating val data', unit = 'data', total = val):
    # use random number of reveals from 0 to 5
    num_reveals = np.random.randint(0, 5)
    input_array, target_array = gen_input_output((10,10), 10, num_reveals = num_reveals)
    np.save(val_path + 'input_' + str(i), input_array)
    np.save(val_path + 'target_' + str(i), target_array)