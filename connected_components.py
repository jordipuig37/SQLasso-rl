# Connected Components functions

def is_valid(self, x, y, key, grid, visited):
    """
    Checks if a cell is valid.
    i.e it is inside the grid and equal to the key
    """
    if self.conf.grid_size > x >= 0 and self.conf.grid_size > y >= 0:
        if visited[x][y] == 0 and grid[x][y] == key:
            return True
        else:
            return False

    else:
        return False


def BFS(self, x, y, i, j, grid, visited, counts):
    """
    BFS to find all cells in connection with key = grid[i][j].
    """
    # global counts

    # terminating case for BFS
    if x != y:
        return

    visited[i][j] = 1
    counts[grid[i][j]] += 1

    # x_move and y_move arrays
    # are the possible movements
    # in x or y direction
    x_move = [0, 0, 1, -1]
    y_move = [1, -1, 0, 0]

    # checks all four points connected with grid[i][j]
    for u in range(4):

        if is_valid(i + y_move[u], j + x_move[u], x, grid, visited):
            BFS(x, y, i + y_move[u], j + x_move[u], grid, visited, counts)


def reset_visited(self, visited):
    """
    Called every time before a BFS so that visited array is reset to zero.
    """
    for i in range(self.conf.grid_size):
        for j in range(self.conf.grid_size):
            visited[i][j] = 0


def get_connected_components(self, grid):
    """
    Computes largest connected component for each player.
    """
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    visited = [[0 for j in range(self.conf.grid_size)] for i in range(self.conf.grid_size)]
    max_size = {1: -1, 2: -1, 3: -1, 4: -1}

    for i in range(self.conf.grid_size):
        for j in range(self.conf.grid_size):
            if grid[i][j] != 0:  # cell is not empty
                reset_visited(visited)
                counts[grid[i][j]] = 0

                # checking cell to the right
                if j + 1 < self.conf.grid_size:
                    BFS(grid[i][j], grid[i][j + 1], i, j, grid, visited, counts)

                # updating result
                if counts[grid[i][j]] >= max_size[grid[i][j]]:
                    max_size[grid[i][j]] = counts[grid[i][j]]

                reset_visited(visited)
                counts[grid[i][j]] = 0

                # checking cell downwards
                if i + 1 < self.conf.grid_size:
                    BFS(grid[i][j], grid[i + 1][j], i, j, grid, visited, counts)

                # updating result
                if counts[grid[i][j]] >= max_size[grid[i][j]]:
                    max_size[grid[i][j]] = counts[grid[i][j]]

    return max_size
