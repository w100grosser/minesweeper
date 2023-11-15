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
hidden_dim = 2048
output_dim = shape[0] * shape[1]
learning_rate = 10
num_mines = 15

input_files = sorted(glob.glob(os.path.join("dataset", 'input_*.npy')))
target_files = sorted(glob.glob(os.path.join("dataset", 'target_*.npy')))

assert len(input_files) == len(target_files)

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

def binary_crossentropy(predictions, targets):
    epsilon = 1e-15  # To prevent log(0)
    predictions = anp.clip(predictions, epsilon, 1 - epsilon)
    loss = -anp.mean(targets * anp.log(predictions) + (1 - targets) * anp.log(1 - predictions))
    return loss

# Neural Network forward pass
def neural_network(X, W1, b1, W2, b2):
    X = X.flatten()
    z1 = anp.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = anp.dot(a1, W2) + b2
    return sigmoid(z2)

def loss(W1, b1, W2, b2, X, Y):
    X = X.flatten()
    Y = Y.flatten()
    predictions = neural_network(X, W1, b1, W2, b2)
    return binary_crossentropy(predictions, Y)

# Compute gradient
grad_W1 = grad(loss, 0)
grad_b1 = grad(loss, 1)
grad_W2 = grad(loss, 2)
grad_b2 = grad(loss, 3)

loss_total = 0

# Training loop
for epoch in range(epochs):
    for batch in tqdm(range(len(input_files)//batch_size), desc=f"Epoch {epoch}", total=len(input_files)//batch_size):
        dW1 = 0
        db1 = 0
        dW2 = 0
        db2 = 0
        batch_loss = 0
        for input_file, target_file in zip(input_files[batch*batch_size: (batch + 1)*batch_size ], target_files[batch*batch_size: (batch + 1)*batch_size ]):
            input_data, output_data = np.load(input_file), np.load(target_file)

            dW1 += grad_W1(W1, b1, W2, b2, input_data, output_data)
            db1 += grad_b1(W1, b1, W2, b2, input_data, output_data)
            dW2 += grad_W2(W1, b1, W2, b2, input_data, output_data)
            db2 += grad_b2(W1, b1, W2, b2, input_data, output_data)

            batch_loss += loss(W1, b1, W2, b2, input_data, output_data)
        
        loss_total += batch_loss/batch_size
        W1 -= learning_rate * dW1 / batch_size
        b1 -= learning_rate * db1 / batch_size
        W2 -= learning_rate * dW2 / batch_size
        b2 -= learning_rate * db2 / batch_size
        
        # calculate running average loss
        if batch == 0:
            running_loss = batch_loss/batch_size
        else:
            running_loss = loss_total / batch
        
        if (batch) % 50 == 0:
            tqdm.write(f"Epoch {epoch}, Batch: {batch}, Loss: {running_loss}")

    if (epoch) % 2 == 0:

        # Create a directory for this epoch
        epoch_dir = os.path.join(base_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir)

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