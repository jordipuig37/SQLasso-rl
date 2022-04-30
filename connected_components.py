# Connected Components functions

def is_valid(x, y, key, grid, visited):
    """
    Checks if a cell is valid.
    i.e it is inside the grid and equal to the key
    """
    grid_size = len(grid)
    if grid_size > x >= 0 and grid_size > y >= 0:
        if visited[x][y] == 0 and grid[x][y] == key:
            return True
        else:
            return False

    else:
        return False


def BFS(x, y, i, j, grid, visited, counts):
    """
    BFS to find all cells in connection with key = grid[i][j].
    """
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


def get_connected_components(grid):
    """
    Computes largest connected component for each player.
    """
    grid_size = len(grid)
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    visited = [[0 for j in range(grid_size)] for i in range(grid_size)]
    max_size = {1: -1, 2: -1, 3: -1, 4: -1}

    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i][j] != 0:  # cell is not empty
                counts[grid[i][j]] = 0

                # checking cell to the right
                if j + 1 < grid_size:
                    BFS(grid[i][j], grid[i][j + 1], i, j, grid, visited, counts)

                # updating result
                if counts[grid[i][j]] >= max_size[grid[i][j]]:
                    max_size[grid[i][j]] = counts[grid[i][j]]

                counts[grid[i][j]] = 0

                # checking cell downwards
                if i + 1 < grid_size:
                    BFS(grid[i][j], grid[i + 1][j], i, j, grid, visited, counts)

                # updating result
                if counts[grid[i][j]] >= max_size[grid[i][j]]:
                    max_size[grid[i][j]] = counts[grid[i][j]]

    return max_size
