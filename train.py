import numpy as np
import autograd.numpy as anp
from autograd import grad
from data_gen import gen_input_output
import random
import os

base_dir = 'saved_models'

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

# Neural Network parameters
shape = (10, 10)
input_dim = 2 * shape[0] * shape[1]
hidden_dim = 128
output_dim = shape[0] * shape[1]
learning_rate = 0.001
epochs = 1000000
num_mines = 15

# Initialize weights
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

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

# Loss function
# def loss(W1, b1, W2, b2, X, Y):
#     X = X.flatten()
#     Y = Y.flatten()
#     predictions = neural_network(X, W1, b1, W2, b2)
#     return anp.mean((predictions - Y)**2)

def loss(W1, b1, W2, b2, X, Y):
    X = X.flatten()
    Y = Y.flatten()
    predictions = neural_network(X, W1, b1, W2, b2)
    return binary_crossentropy(predictions, Y)

# Compute gradient
# loss_gradient = grad(loss, argnums=[0, 1, 2, 3])
grad_W1 = grad(loss, 0)
grad_b1 = grad(loss, 1)
grad_W2 = grad(loss, 2)
grad_b2 = grad(loss, 3)

# Training loop
for epoch in range(epochs):
    input_data, output_data = gen_input_output(shape, num_mines, num_reveals = random.randint(1, 5))

    dW1 = grad_W1(W1, b1, W2, b2, input_data, output_data)
    db1 = grad_b1(W1, b1, W2, b2, input_data, output_data)
    dW2 = grad_W2(W1, b1, W2, b2, input_data, output_data)
    db2 = grad_b2(W1, b1, W2, b2, input_data, output_data)
    
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss(W1, b1, W2, b2, input_data, output_data)}")

    if (epoch + 1) % 20000 == 0:
        # Create a directory for this epoch
        epoch_dir = os.path.join(base_dir, f'epoch_{epoch+1}')
        os.makedirs(epoch_dir)

        # Save the weights in this directory
        np.save(os.path.join(epoch_dir, 'W1.npy'), W1)
        np.save(os.path.join(epoch_dir, 'b1.npy'), b1)
        np.save(os.path.join(epoch_dir, 'W2.npy'), W2)
        np.save(os.path.join(epoch_dir, 'b2.npy'), b2)
        print(f"Saved model weights at epoch {epoch+1} in {epoch_dir}")

# Save the weights after training
np.save('W1.npy', W1)
np.save('b1.npy', b1)
np.save('W2.npy', W2)
np.save('b2.npy', b2)

# Prediction
def predict(X):
    return neural_network(X, W1, b1, W2, b2)
