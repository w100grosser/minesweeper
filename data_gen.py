from engine import Minesweeper
import numpy as np
from scipy.signal import convolve2d

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

    # print(engine.revealed)
    # print(get_border(engine))
    # print(border_mines)

    # print(engine.reveal_random_more(1))
    # print(engine.revealed)

    # target_array = get_border(engine)
    target_array = np.where(border_mines == 1, 1, 0)
    # print(target_array)

    input_array = np.zeros((2,*shape))

    # print(input_array.shape)

    input_array[0] = 1 - engine.revealed

    revealed_numbers = np.logical_and(engine.revealed, engine.board >= 0).astype(int)

    input_array[1] = np.where(revealed_numbers, engine.board/8, 0)

    # print(input_array)
    # print(target_array)
    return input_array, target_array

shape = (10, 10)

input_array, target_array = gen_input_output(shape, 10, 40)

# print(input_array)
# print(target_array)