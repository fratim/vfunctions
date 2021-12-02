import numpy as np
from dijkstar import Graph, find_path

from parameters import FNAME_PLAN, GAMMA, N_GOALS
import matplotlib.pyplot as plt
from scipy import signal
from skimage.feature import peak_local_max


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
            # add bottom right edge if not outside
            if ((i + 1) < grid.shape[0] and (j + 1) < grid.shape[1]
                    and grid[i + 1, j + 1] == 0):
                graph.add_edge(node_id, node_id + grid.shape[1] + 1)

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
    generated_goals = list()
    while (len(generated_goals) < n_goals):
        i = np.random.randint(0, env_grid.shape[0])
        j = np.random.randint(0, env_grid.shape[1])

        coods = (i, j)
        if coods not in generated_goals and env_grid[coods] == 0:
            generated_goals.append(coods)

    return generated_goals


def get_node_value(graph, source_node_id, goal_node_id, gamma):
    try:
        return gamma**sum(find_path(graph, source_node_id, goal_node_id, cost_func=dummy_cost_func).costs)
        #return sum(find_path(graph, source_node_id, goal_node_id, cost_func=dummy_cost_func).costs)
    except:
        return 0
def get_value_functions(env_grid, env_graph, generated_goals, gamma):
    """
    generate 3D array that hold value functions for all n goals in generated goals list
    :param env_grid:
    :param generated_goals:
    :return: 3D grid that contains all value functions
    """

    n_goals = len(generated_goals)

    value_fcts = np.ones((env_grid.shape + (n_goals,))) * 0

    for g, goal in enumerate(generated_goals):

        goal = generated_goals[g]
        goal_node_id = goal[0]*env_grid.shape[1] + goal[1]

        node_id = 0
        for i in range(env_grid.shape[0]):
            for j in range(env_grid.shape[1]):
                value_fcts[i, j, g] = get_node_value(env_graph, node_id, goal_node_id, gamma)
                node_id += 1

    return value_fcts


def main():

    env_grid = read_input_grid_from_file(FNAME_PLAN)
    env_graph = convert_grid_to_graph(env_grid)
    generated_goals = generate_random_goals(env_grid, N_GOALS)
    value_functions = get_value_functions(env_grid, env_graph, generated_goals, GAMMA)

    # for g in range(len(generated_goals)):
    #     plt.imshow(value_functions[:, :, g])
    #     plt.show()

    averaged_values = np.average(value_functions, axis=2)
    plt.imshow(averaged_values)

    for goal in generated_goals:
        plt.plot(goal[1], goal[0], marker='.', color="red")

    maximums = peak_local_max(averaged_values, exclude_border=False)

    for maximum in maximums:
        plt.plot(maximum[1], maximum[0], marker='v', color="black")

    print("HALLO")




if __name__ == "__main__":
    main()