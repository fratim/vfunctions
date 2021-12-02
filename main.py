import numpy as np
from dijkstar import Graph, find_path

from parameters import FNAME_PLAN
import matplotlib.pyplot as plt


def dummy_cost_func(a, b, c, d):
    return 1

def convert_grid_to_graph(grid):
    """

    :param grid: input 2d numpy array grid
    :return:
    """

    graph = Graph(undirected=True)

    node_id = 0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            # if node is blocked, no connections are established
            if grid[i, j] != 0:
                node_id += 1
                continue

            # add right edge if not last row and not blocked
            if (j + 1) < grid.shape[1] and grid[i, j + 1] == 0:
                graph.add_edge(node_id, node_id + 1)
            # add below edge if not outside of grid and not blocked
            if (i + 1) < grid.shape[0] and grid[i + 1, j] == 0:
                graph.add_edge(node_id, node_id + grid.shape[1])

            print(graph)
            # increase node_counter
            node_id += 1

    return graph



def read_grid_map(grid_map_path):
    with open(grid_map_path, 'r') as f:
        grid_map = f.readlines()

    grid_map_array = np.array(
        list(map(
            lambda x: list(map(
                lambda y: int(y),
                x.split(' ')
            )),
            grid_map
        ))
    )
    return grid_map_array

def read_input_grid_from_file(fname):
    """
    read from text file and return 2D array that is 0 where cells are unoccupied and 1 where occupied
    :param fname: fname
    :return: 2D np array
    """

    return read_grid_map(fname)

def generate_random_goals(env_grid, n_goals):
    """
    generate n random goals within the grid that are given by coordinated of unoccupied cells
    :return: list of n_goals
    """

def get_node_distance(graph, source_node_id, goal_node_id):
    try:
        return sum(find_path(graph, source_node_id, goal_node_id, cost_func=dummy_cost_func).costs)
    except:
        return -1
def get_value_functions(env_grid, env_graph, generated_goals):
    """
    generate 3D array that hold value functions for all n goals in generated goals list
    :param env_grid:
    :param generated_goals:
    :return: 3D grid that contains all value functions
    """

    n_goals = len(generated_goals)

    for g, goal in enumerate(generated_goals):

        goal = generated_goals[0]
        goal_node_id = goal[0]*env_grid.shape[1] + goal[1]

        value_fcts = np.ones((env_grid.shape + (n_goals,)))*-10

        node_id = 0
        for i in range(env_grid.shape[0]):
            for j in range(env_grid.shape[1]):
                value_fcts[i, j, g] = get_node_distance(env_graph, node_id, goal_node_id)
                node_id += 1

        return value_fcts


def main():

    env_grid = read_input_grid_from_file(FNAME_PLAN)
    env_graph = convert_grid_to_graph(env_grid)
    print("HALLO")
    #generated_goals = generate_random_goals(env_grid)
    generated_goals = [(1, 1)]
    value_functions = get_value_functions(env_grid, env_graph, generated_goals)
    print("HALLO")

    plt.imshow(value_functions[:, :, 0])
    plt.show()

    print("HALLO")




if __name__ == "__main__":
    main()