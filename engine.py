import numpy as np
import random
from scipy.signal import convolve2d

class Minesweeper:
    """
    A Minesweeper game engine.
    
    Attributes:
        shape (tuple): The shape of the game board.
        num_mines (int): Total number of mines on the board.
        board (np.ndarray): A matrix representing the game board.
        revealed (np.ndarray): A matrix representing revealed cells.
        mines (set): A set containing coordinates of mines.
        game_over (bool): Indicates if the game is over.
        game_won (bool): Indicates if the game is won.
        num_revealed (int): Number of revealed cells.
        num_flags (int): Number of flagged cells.
        zeros (np.ndarray): A matrix identifying zeros on the board.
    """


    def __init__(self, shape, num_mines):
        """
        Initializes a new Minesweeper game.
        
        Args:
            shape (tuple): Shape of the game board (rows, columns).
            num_mines (int): Number of mines to be placed on the board.
        """
        self.shape = shape
        self.num_mines = num_mines
        self.board = np.zeros(shape, dtype=int)
        self.revealed = np.zeros(shape, dtype=int)
        self.mines = set()
        self.game_over = False
        self.game_won = False
        self.num_revealed = 0
        self.num_flags = 0
        self.initialize_board()
        self.zeros = np.where(self.board == 0, 1, 0)

    def initialize_board(self):
        """Initializes the board with mines and computes the numbers for surrounding cells."""
        indices = [(i, j) for i in range(self.shape[0]) for j in range(self.shape[1])]
        mine_indices = random.sample(indices, self.num_mines)
        for i, j in mine_indices:
            self.board[i][j] = -1
            self.mines.add((i, j))
            for ii in range(i-1, i+2):
                for jj in range(j-1, j+2):
                    if ii >= 0 and ii < self.shape[0] and jj >= 0 and jj < self.shape[1] and self.board[ii][jj] != -1:
                        self.board[ii][jj] += 1

    def reveal(self, i, j):
        """
        Reveals a cell and its surroundings if it's an empty cell.
        
        Args:
            i (int): Row index of the cell to reveal.
            j (int): Column index of the cell to reveal.
        """
        if (i, j) in self.mines:
            self.game_over = True
            return
        
        if self.board[i][j] != 0:
            self.revealed[i][j] = True
            return

        queue = [(i, j)]
        queue_discovered = []

        while queue:
            (i,j) = queue.pop(0)
            for ii in range(max(i-1, 0), min(i+1, self.shape[0]-1) + 1):
                for jj in range(max(j-1, 0), min(j+1, self.shape[1]-1) + 1):
                    if self.board[ii,jj] >= 0 and self.board[ii,jj] <= 8 and (ii,jj) not in queue and (ii,jj) not in queue_discovered:
                        if self.board[ii,jj] == 0:
                            queue.append((ii,jj))
                        self.revealed[ii][jj] = True
            queue_discovered.append((i,j))
            
    def reveal_random (self):
        """Reveals a random cell that has a value of 0."""
        if self.game_over or self.game_won:
            self.game_over = True
            return
        

        indicies = []
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.board[i][j] == 0:
                    indicies.append((i,j))

        random_index = random.choice(indicies)

        i,j = random_index

        # print(i,j)

        if self.game_over or self.game_won or (i, j) in self.mines:
            self.game_over = True
            return
        
        self.reveal(i,j)
        return random_index
    
    def reveal_random_more(self, num_reveals):
        """
        Reveals multiple random border cells.
        
        Args:
            num_reveals (int): Number of random border cells to reveal.
        """
        for i in range(num_reveals):

            indicies = []
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if not (self.board[i][j] in [-3, -1]) and self.get_border()[i][j] == 1:
                        indicies.append((i,j))
            if indicies == []:
                return
            random_index = random.choice(indicies)

            i,j = random_index

            # print(i,j)

            if self.game_over or self.game_won or (i, j) in self.mines:
                self.game_over = True
                return

            self.reveal(i,j)

        return random_index

    def flag(self, i, j):
        """
        Flags a cell.
        
        Args:
            i (int): Row index of the cell to flag.
            j (int): Column index of the cell to flag.
        """
        if self.board[i][j] >= 0:
            self.board[i][j] = -3
            self.num_flags += 1

    def unflag(self, i, j):
        """
        Unflags a cell.
        
        Args:
            i (int): Row index of the cell to unflag.
            j (int): Column index of the cell to unflag.
        """
        if self.board[i][j] == -3:
            self.board[i][j] = 0
            self.num_flags -= 1

    def get_game_status(self):
        """
        Gets the current game status.

        Returns:
            str: "lost" if game is lost, "won" if game is won, "ongoing" otherwise.
        """
        if self.game_over:
            return "lost"
        elif self.game_won:
            return "won"
        else:
            return "ongoing"

    def print_board(self):
        """Prints the current state of the game board."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.board[i][j] == -2:
                    print(".", end=" ")
                elif self.board[i][j] == -3:
                    print("F", end=" ")
                else:
                    print(" ", end=" ")
            print()

    def get_board(self):
        """Returns the current state of the game board."""
        return self.board

    def get_revealed(self):
        """Returns the current state of revealed cells."""
        return self.revealed

    def get_border(self):
        """
        Computes the border cells that are adjacent to revealed cells.

        Returns:
            np.ndarray: A matrix indicating border cells.
        """
        kernel = np.ones((3, 3))
        convolved = convolve2d(self.revealed, kernel, mode='same')
        result = np.logical_and(convolved > 0, self.revealed == 0).astype(int)
        return result

# The values of the board are as follows:
# -1: mine
# -2: revealed
# -3: flagged
# -4: hidden
# 0-8: number of mines in the surrounding cells


# Minesweeper class documentation for someone to use it in their code
# Minesweeper(shape, num_mines): creates a Minesweeper object with the given shape and number of mines
# reveal(i, j): reveals the cell at (i, j) and all the surrounding cells if the cell is empty
# flag(i, j): flags the cell at (i, j)
# unflag(i, j): unflags the cell at (i, j)
# get_board(): returns the current board
# get_game_status(): returns the current game status
# print_board(): prints the current board
