import numpy as np
import autograd.numpy as anp
from autograd import grad
from data_gen import gen_input_output
import random

# Neural Network parameters
shape = (10, 10)
input_dim = 2 * shape[0] * shape[1]
hidden_dim = 128
output_dim = shape[0] * shape[1]
learning_rate = 0.001
epochs = 10000
num_mines = 15


# Save the weights after training
W1 = np.load('W1.npy')
b1 = np.load('b1.npy')
W2 = np.load('W2.npy')
b2 = np.load('b2.npy')

# Activation function
def sigmoid(x):
    return 1 / (1 + anp.exp(-x))

# Neural Network forward pass
def neural_network(X, W1, b1, W2, b2):
    X = X.flatten()
    z1 = anp.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = anp.dot(a1, W2) + b2
    return sigmoid(z2)



# Prediction
def predict(X):
    return neural_network(X, W1, b1, W2, b2)

input_data, output_data = gen_input_output(shape, num_mines, num_reveals = random.randint(1, 5))


predictions = predict(input_data.flatten())
print(output_data)
print((predictions.reshape(shape)*100).astype(int))
