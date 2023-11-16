import numpy as np
import autograd.numpy as anp
from autograd import grad
from data_gen import gen_input_output
import random
import os
from tqdm import tqdm
import glob

base_dir = 'saved_models'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Neural Network parameters
shape = (10, 10)
batch_size = 32
epochs = 10
input_dim = 2 * shape[0] * shape[1]
hidden_dim = 512
output_dim = shape[0] * shape[1]
learning_rate = 20
num_mines = 15

train_input_files = sorted(glob.glob(os.path.join("datasetnewbatch/train", 'input_*.npy')))
train_target_files = sorted(glob.glob(os.path.join("datasetnewbatch/train", 'target_*.npy')))
assert len(train_target_files) == len(train_input_files)

test_input_files = sorted(glob.glob(os.path.join("datasetnewbatch/test", 'input_*.npy')))
test_target_files = sorted(glob.glob(os.path.join("datasetnewbatch/test", 'target_*.npy')))
assert len(test_input_files) == len(test_target_files)

val_input_files = sorted(glob.glob(os.path.join("datasetnewbatch/val", 'input_*.npy')))
val_target_files = sorted(glob.glob(os.path.join("datasetnewbatch/val", 'target_*.npy')))
assert len(val_input_files) == len(val_target_files)

# Initialize weights
W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
b2 = np.zeros((1, output_dim))

# load weights if they exist
# if os.path.exists('W1.npy'):
#     W1 = np.load('W1.npy')
#     b1 = np.load('b1.npy')
#     W2 = np.load('W2.npy')
#     b2 = np.load('b2.npy')

# Activation function
def sigmoid(x):
    return 1 / (1 + anp.exp(-x))

# Activation function
def relu(x):
    return anp.maximum(0, x)

def binary_crossentropy(predictions, targets):
    epsilon = 1e-15  # To prevent log(0)
    predictions = anp.clip(predictions, epsilon, 1 - epsilon)
    loss = -anp.mean(targets * anp.log(predictions) + (1 - targets) * anp.log(1 - predictions))
    return loss

# Neural Network forward pass
def neural_network(X, W1, b1, W2, b2):
    z1 = anp.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = anp.dot(a1, W2) + b2
    return sigmoid(z2)

def loss(W1, b1, W2, b2, X, Y):
    predictions = neural_network(X, W1, b1, W2, b2)
    return binary_crossentropy(predictions, Y)

# Compute gradient
grad_W1 = grad(loss, 0)
grad_b1 = grad(loss, 1)
grad_W2 = grad(loss, 2)
grad_b2 = grad(loss, 3)


# Training loop
for epoch in range(epochs):
    loss_total = 0
    for batch, (input_file, target_file) in tqdm(enumerate(zip(train_input_files, train_target_files)), desc=f"Epoch {epoch}", total=len(train_input_files)):

    
        input_data, output_data = np.load(input_file), np.load(target_file)
        # print(X_batch.shape, Y_batch.shape)

        dW1 = grad_W1(W1, b1, W2, b2, input_data, output_data)
        db1 = grad_b1(W1, b1, W2, b2, input_data, output_data)
        dW2 = grad_W2(W1, b1, W2, b2, input_data, output_data)
        db2 = grad_b2(W1, b1, W2, b2, input_data, output_data)

        batch_loss = loss(W1, b1, W2, b2, input_data, output_data)
        
        loss_total += batch_loss
        W1 -= learning_rate * dW1 / batch_size
        b1 -= learning_rate * db1 / batch_size
        W2 -= learning_rate * dW2 / batch_size
        b2 -= learning_rate * db2 / batch_size
        
        # calculate running average loss
        if batch == 0:
            running_loss = batch_loss/batch_size
        else:
            running_loss = loss_total / batch
        
        if (batch) % 50 == 0 and batch > 0:
            tqdm.write(f"Epoch {epoch}, Batch: {batch}, Loss: {running_loss}")

    # try out the model on the test set
    test_loss = 0
    for input_file, target_file in tqdm(zip(test_input_files, test_target_files), desc="Testing", total=len(test_input_files)):
        input_data, output_data = np.load(input_file), np.load(target_file)
        test_loss += loss(W1, b1, W2, b2, input_data, output_data)
    test_loss /= len(test_input_files)
    tqdm.write(f"Test Loss: {test_loss}")

    if (epoch) % 2 == 0:

        # Create a directory for this epoch
        epoch_dir = os.path.join(base_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)

        # Save the weights in this directory
        np.save(os.path.join(epoch_dir, 'W1.npy'), W1)
        np.save(os.path.join(epoch_dir, 'b1.npy'), b1)
        np.save(os.path.join(epoch_dir, 'W2.npy'), W2)
        np.save(os.path.join(epoch_dir, 'b2.npy'), b2)
        tqdm.write(f"Saved model weights at epoch {epoch+1} in {epoch_dir}")

# Save the weights after training
np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)