import numpy as np
from dijkstar import Graph, find_path
import copy

from parameters import *
import matplotlib.pyplot as plt
from scipy.special import softmax
from skimage.feature import peak_local_max


def dummy_cost_func(a, b, c, d):
    return 1

ACTIONID_TO_ACTION_DICT = {
    "0": ".",
    "1": ">",
    "2": "v",
    "3": "<",
    "4": "^",
}

def g_node_id(row, col, n_columns):
    return row*n_columns + col


def convert_grid_to_graph(grid, diagonal_movements_possible=False):
    """

    :param grid: input 2d numpy array grid
    :return:
    """

    graph = Graph(undirected=True)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):

            # if node is blocked, no connections are established
            if grid[i, j] != 0:
                continue

            node_id = g_node_id(i, j, grid.shape[1])

            # add right edge if not last row and not blocked
            if (j + 1) < grid.shape[1] and grid[i, j + 1] == 0:
                graph.add_edge(node_id, node_id + 1)

            # add below edge if not outside of grid and not blocked
            if (i + 1) < grid.shape[0] and grid[i + 1, j] == 0:
                graph.add_edge(node_id, node_id + grid.shape[1])

            if diagonal_movements_possible:
                # # add bottom right edge if not outside
                if ((i + 1) < grid.shape[0] and (j + 1) < grid.shape[1]
                        and grid[i + 1, j + 1] == 0):
                    graph.add_edge(node_id, node_id + grid.shape[1] + 1)

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


def generate_random_goals(env_grid, n_goals, row_limits=None, column_limits=None):
    """
    generate n random goals within the grid that are given by coordinated of unoccupied cells
    :return: list of n_goals
    """
    if row_limits is None:
        row_limits = (0, env_grid.shape[0])
    if column_limits is None:
        column_limits = (0, env_grid.shape[1])

    generated_goals = list()

    if n_goals == "ALL":
        for i in range(env_grid.shape[0]):
            for j in range(env_grid.shape[1]):
                coods = (i, j)
                if env_grid[coods] == 0:
                    generated_goals.append((i, j))
        return generated_goals


    while (len(generated_goals) < n_goals):
        i = np.random.randint(row_limits[0], row_limits[1])
        j = np.random.randint(column_limits[0], column_limits[1])

        coods = (i, j)
        if coods not in generated_goals and env_grid[coods] == 0:
            generated_goals.append(coods)

    return generated_goals


def get_node_value(graph, source_node_id, goal_node_id, gamma):
    """
    returns the value function value for a node, which is computed using the discount factor and the distance to goal
    :param graph:
    :param source_node_id:
    :param goal_node_id:
    :param gamma:
    :return:
    """
    distance = get_distance(graph, source_node_id, goal_node_id)
    return gamma**distance


def get_distance(graph, source_node_id, goal_node_id):
    """
    get distance between source node and goal node
    if except, no path can be found and large distance is returned
    :param graph:
    :param source_node_id:
    :param goal_node_id:
    :return:
    """
    try:
        return sum(find_path(graph, source_node_id, goal_node_id, cost_func=dummy_cost_func).costs)
    except:
        return NODE_DISTANCE_IF_NO_PATH


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

        print(f"at {g} of {n_goals}")

        goal = generated_goals[g]
        goal_node_id = g_node_id(*goal, env_grid.shape[1])

        for i in range(env_grid.shape[0]):
            for j in range(env_grid.shape[1]):
                node_id = g_node_id(i, j, env_grid.shape[1])
                value_fcts[i, j, g] = get_node_value(env_graph, node_id, goal_node_id, gamma)

    return value_fcts


def get_policies(env_grid, env_graph, generated_goals):
    """
    generate 3D array that holds policy for all n goals in generated goals list
    :param env_grid:
    :param generated_goals:
    :return: 3D grid that contains all value functions
    """

    n_goals = len(generated_goals)

    policies = np.ones((env_grid.shape + (n_goals, 5))) * np.nan

    for g, goal in enumerate(generated_goals):

        print(f"at {g} of {n_goals}")

        goal = generated_goals[g]
        goal_node_id = goal[0]*env_grid.shape[1] + goal[1]

        node_id = 0
        for i in range(env_grid.shape[0]):
            for j in range(env_grid.shape[1]):
                nb_distances = get_neighboring_distances(env_grid, env_graph, i, j, node_id, goal_node_id)
                policy = softmax(SOFTMAX_MULTIPLIER*1/(nb_distances+SOFTMAX_EPSILON))
                policies[i, j, g, :] = policy
                node_id += 1

    return policies


def get_neighboring_distances(env_grid, env_graph, i, j, node_id, goal_node_id):
    """
    get the distance to goal for all neighboring cells of a specific cell
    this can then be used to find the shortest-path action to the goal from a given cell
    :param env_grid:
    :param env_graph:
    :param i:
    :param j:
    :param node_id:
    :param goal_node_id:
    :return:
    """
    pb_actions = [0, 1, 2, 3, 4] # put, right, down, left, up
    goal_distances = np.ones((1, len(pb_actions)))*np.nan

    # add stay distance
    goal_distances[0, 0] = get_distance(env_graph, node_id, goal_node_id)

    # add right cell distance
    if j < (env_grid.shape[1]-1):
        env_right_node_id = node_id + 1
        goal_distances[0, 1] = get_distance(env_graph, env_right_node_id, goal_node_id)
    else:
        goal_distances[0, 1] = np.inf

    # add bottom cell distance
    if i < (env_grid.shape[0]-1):
        env_bottom_node_id = node_id + env_grid.shape[1]
        goal_distances[0, 2] = get_distance(env_graph, env_bottom_node_id, goal_node_id)
    else:
        goal_distances[0, 2] = np.inf

    # add left cell distance
    if j > 0:
        env_left_node_id = node_id - 1
        goal_distances[0, 3] = get_distance(env_graph, env_left_node_id, goal_node_id)
    else:
        goal_distances[0, 3] = np.inf

    # add top cell distance
    if i > 0:
        env_top_node_id = node_id - env_grid.shape[1]
        goal_distances[0, 4] = get_distance(env_graph, env_top_node_id, goal_node_id)
    else:
        goal_distances[0, 4] = np.inf

    # best_action = np.argwhere(goal_distances==np.min(goal_distances))[0][0]

    return goal_distances


def plot_optimal_actions(rows, cols, goal, distances, offset, color):
    """
    plots action that is chosen with highest probability
    :param rows:
    :param cols:
    :param goal:
    :param distances:
    :param offset:
    :param color:
    :return:
    """
    for i in range(rows):
        for j in range(cols):

            nb_distances = np.around(distances[i, j, goal, :] , 4)
            best_actions = np.argwhere(nb_distances == np.max(nb_distances))
            best_action_icons = []
            for action in best_actions:
                best_action_icons.append(ACTIONID_TO_ACTION_DICT[str(action[0])])

            best_action_icons = ' '.join(best_action_icons)
            plt.text(j, i+offset, best_action_icons, color=color, ha='center', va='center')


def compute_weighted_average_policies(policies, generated_goals, env_graph, env_grid):

    avg_policies = np.zeros((env_grid.shape)+(1, policies.shape[-1], ))

    for i in range(policies.shape[0]):
        for j in range(policies.shape[1]):

            curr_node_id = i*env_grid.shape[1] + j

            state_policies = policies[i, j, :, :]
            goal_distances = np.ones((1, len(generated_goals)))*np.nan

            for goal_id, goal in enumerate(generated_goals):
                goal_node_id = g_node_id(*goal, env_grid.shape[1])
                goal_distances[0, goal_id] = get_distance(env_graph, curr_node_id, goal_node_id)

            # p(g|s) assumed to be proportional to GOAL_ASSIGNMENT_DECAY_FACTOR^(distance_to_g)
            decay_factor = GOAL_ASSIGNMENT_DECAY_FACTOR
            goal_probabilities = np.ones((1, len(generated_goals)))*decay_factor
            goal_probabilities = goal_probabilities**goal_distances

            for goal_id in range(len(generated_goals)):
                avg_policies[i, j, 0, :] += goal_probabilities[0, goal_id]*state_policies[goal_id, :]

            avg_policies[i, j, 0, :] = avg_policies[i, j, 0, :]/np.sum(avg_policies[i, j, 0, :])

    return avg_policies

def colorplot_values(ax, values):
    ax.imshow(values)

def plot_optimal_action_for_all_goals(ax, policies, init_offset):
    colors = ["yellow", "red", "blue", "green"]
    offset = copy.deepcopy(init_offset)
    for g in range(policies.shape[2]):
        plot_optimal_actions(policies.shape[0], policies.shape[1], g, policies, offset, colors[g])
        offset += 0.15

    return offset

def plot_optimal_action_for_average_policy(ax, avg_policy, curr_offset):
    plot_optimal_actions(avg_policy.shape[0], avg_policy.shape[1], 0, avg_policy, curr_offset, "black")

def plot_goal_positions(ax, goals):
    for goal in goals:
        ax.plot(goal[1], goal[0], marker='.', color="red")


def plot_local_value_maxima(avg_values, goals):

    maxima = peak_local_max(avg_values, exclude_border=False)

    for maximum in maxima:
        if tuple((maximum[0], maximum[1])) in goals and N_GOALS!= "ALL":
            color = "red"
        else:
            color = "black"

        plt.plot(maximum[1], maximum[0], marker='v', color=color)


def make_plots(value_functions, avg_values, policies, avg_policies, goals):
    fig, ax = plt.subplots()

    # colorplot mean values
    colorplot_values(ax, avg_values)

    # plot optimal actions for all goals
    curr_offset = plot_optimal_action_for_all_goals(ax, policies, init_offset=-0.3)

    # plot optimal action for average policy underneath in black
    curr_offset = plot_optimal_action_for_average_policy(ax, avg_policies, curr_offset)

    # plot positions of all goals in red
    # TODO change colors to respective goal policy colors
    plot_goal_positions(ax, goals)

    # plot local maxima in value function
    plot_local_value_maxima(avg_values, goals)

    fig.savefig("plot.png")


def main():

    # read grid and convert to graph
    env_grid = read_input_grid_from_file(FNAME_PLAN)
    env_graph = convert_grid_to_graph(env_grid)

    # generate or specify goals
    #generated_goals = generate_random_goals(env_grid, N_GOALS, ROW_LIMITS, COLUMN_LIMITS)
    horiz_center = int(env_grid.shape[1]/2)
    generated_goals = [(0, horiz_center), (2, 0), (2, env_grid.shape[1]-1), (5, horiz_center)]

    # compute value functions for all goals
    value_functions = get_value_functions(env_grid, env_graph, generated_goals, GAMMA)


    # compute policies for all goals
    policies = get_policies(env_grid, env_graph, generated_goals)
    avg_values = np.mean(value_functions, axis=2)

    # compute weighted average policy
    avg_policy = compute_weighted_average_policies(policies, generated_goals, env_graph, env_grid)

    # make various plots
    make_plots(value_functions, avg_values, policies, avg_policy, generated_goals)

if __name__ == "__main__":
    main()