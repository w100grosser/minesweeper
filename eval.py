from engine import Minesweeper
import numpy as np
import autograd.numpy as anp
from scipy.signal import convolve2d
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import time
import os
import wandb


def visualize_board(game, flag=np.zeros((10,10))):
    """
    Visualizes the Minesweeper board using provided images.

    :param game: Minesweeper game object
    :return: None
    """
    # Mapping each cell value to its corresponding image
    image_dict = {
        -1: 'mine.png',
        -3: 'flag.png',
        -4: 'covered.png'
    }
    
    board = game.board.copy()
    board[flag == 1] = -3

    for i in range(9):  # 0-8 for the number of mines in the surrounding cells
        image_dict[i] = f'{i}.png'
    fig, ax = plt.subplots(figsize=(10, 10))

    # Go through each cell and display its corresponding image
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            # If the cell is revealed
            if game.revealed[i][j] == 1:
                cell_value = board[i][j]
            else:
                if board[i][j] == -3:
                    cell_value = -3
                else:
                    cell_value = -4  # Covered/Hidden
            ax.imshow(mpimg.imread(f'images/{image_dict[cell_value]}'), extent=(j, j+1, board.shape[0]-i-1, board.shape[0]-i))

    ax.set_xticks(range(board.shape[1]))
    ax.set_yticks(range(board.shape[0]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which='both')
    plt.savefig(f'minesweeper_board.png')
    plt.close(fig)  # Close the plot to free up memory
    
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
    X = X.flatten()
    z1 = anp.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = anp.dot(a1, W2) + b2
    return sigmoid(z2)
    
def gen_input_output(shape, num_mines, num_reveals = 0):

    engine = Minesweeper(shape, num_mines)

    i, j = engine.reveal_random()
    engine.reveal_random_more(num_reveals)

    def get_border(engine):
        kernel = np.ones((3, 3))
        convolved = convolve2d(engine.revealed, kernel, mode='same')
        result = np.logical_and(convolved > 0, engine.revealed == 0).astype(int)
        return result

    border_mines = np.logical_and(get_border(engine), engine.board == -1).astype(int)

    target_array = np.where(border_mines == 1, 1, 0)

    input_array = np.zeros((2,*shape))

    input_array[0] = 1 - engine.revealed

    revealed_numbers = np.logical_and(engine.revealed, engine.board >= 0).astype(int)

    input_array[1] = np.where(revealed_numbers, engine.board/8, 0)

    return input_array, target_array, engine
    
if __name__ == '__main__':
    
    W1 = np.load('W1.npy')
    b1 = np.load('b1.npy')
    W2 = np.load('W2.npy')
    b2 = np.load('b2.npy')

    input_array, target_array, game = gen_input_output((10,10), 10, num_reveals = 2)
    
    # Forward pass
    pred = neural_network(input_array.flatten(), W1, b1, W2, b2)
    pred = (pred >= 0.5).astype(int)
    
    # reshape pred to match input_array
    pred = pred.reshape(target_array.shape)
    
    # plot these three side by side
    fig, ax = plt.subplots(nrows = 1, ncols = 3, figsize=(20, 10))

    visualize_board(game)
    ax[0].imshow(plt.imread('minesweeper_board.png'))
    ax[0].set_title("Original Game")    

    visualize_board(game, flag=pred)
    ax[1].imshow(plt.imread('minesweeper_board.png'))
    ax[1].set_title("Predicton")    

    visualize_board(game, flag=target_array)
    ax[2].imshow(plt.imread('minesweeper_board.png'))
    ax[2].set_title("Ground Truth")    
    
    plt.show()
