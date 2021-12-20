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

PB_ACTIONS = [0, 1, 2, 3, 4]  # put, right, down, left, up



class Grid:

    def __init__(self, fname):
        self.grid = self.read_from_map(fname)

        self.nrows = self.grid.shape[0]
        self.ncols = self.grid.shape[1]

        self.all_coords = self.get_all_coords()

        self.graph = None


    def __getitem__(self, arg):
        if len(arg) == 1:
            return self.grid[self.coords_from_id(arg)]
        elif len(arg) == 2:
            return self.grid[arg]
        else:
            raise ValueError


    def get_all_coords(self):
        all_coords = []
        for i in range(self.nrows):
            for j in range(self.ncols):
                all_coords.append((i, j))

        return all_coords

    def read_from_map(self, fname):
        with open(fname, 'r') as f:
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


    def coords_from_id(self, node_id):
        out_j = node_id % self.ncols
        out_i = node_id // self.ncols

        return (out_i, out_j)


    def id_from_coords(self, coords):
        return coords[0] * self.ncols + coords[1]


    def is_free(self, node_id):
        if self.grid[self.coords_from_id(node_id)] == 0:
            return True
        else:
            return False


    def get_neighbor_node_id(self, input, direction):

        if isinstance(input, int):
            input_coords = self.coords_from_id(input)
        elif isinstance(input, tuple):
            input_coords = input
        else:
            raise NotImplementedError

        input_i = input_coords[0]
        input_j = input_coords[1]

        if direction is None:
            return self.id_from_coords((input_i, input_j))
        elif direction == "east":
            # add right edge if not last row and not blocked
            if not (input_j + 1) < self.ncols:
                return None
            else:
                return self.id_from_coords((input_i, input_j + 1))
        elif direction == "southeast":
            if not (input_i + 1) < self.nrows or not (input_j + 1) < self.ncols:
                return None
            else:
                return self.id_from_coords((input_i + 1, input_j + 1))
        elif direction == "south":
            # add right edge if not last row and not blocked
            if not (input_i + 1) < self.nrows:
                return None
            else:
                return self.id_from_coords((input_i + 1, input_j))
        elif direction == "west":
            if not input_j > 0:
                return None
            else:
                return self.id_from_coords((input_i, input_j - 1))
        elif direction == "north":
            if not input_i > 0:
                return None
            else:
                return self.id_from_coords((input_i - 1, input_j))
        else:
            raise  NotImplementedError("Direction not implemented")


    def convert_to_graph(self, diagonal_movements_possible=False):
        """
        :param grid: input 2d numpy array grid
        :return:
        """

        graph = Graph(undirected=True)

        for coords in self.all_coords:

            node_id = self.id_from_coords(coords)

            # if node is blocked, no connections are established
            if not self.is_free(node_id):
                continue

            directions = ["east", "south"]
            if diagonal_movements_possible:
                directions += ["southwest, southeast"]

            for direction in directions:
                # add east node
                neighboring_node = self.get_neighbor_node_id(node_id, direction)
                if neighboring_node and self.is_free(neighboring_node):
                    graph.add_edge(node_id, neighboring_node)

        self.graph = graph


    def get_distance(self, source_node_id, goal_node_id):
        """
        get distance between source node and goal node
        if except, no path can be found and large distance is returned
        :param graph:
        :param source_node_id:
        :param goal_node_id:
        :return:
        """
        try:
            return sum(find_path(self.graph, source_node_id, goal_node_id, cost_func=dummy_cost_func).costs)
        except:
            return NODE_DISTANCE_IF_NO_PATH


    def get_node_value(self, source_node_id, goal_node_id, gamma):
        """
        returns the value function value for a node, which is computed using the discount factor and the distance to goal
        :param graph:
        :param source_node_id:
        :param goal_node_id:
        :param gamma:
        :return:
        """
        distance = self.get_distance(source_node_id, goal_node_id)
        return gamma ** distance


    def get_neighboring_distances(self, node_id, goal_node_id):
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
        pb_actions = PB_ACTIONS
        goal_distances = np.ones((1, len(pb_actions))) * np.nan

        # add right cell distance
        directions = [None, "east", "south", "west", "north"]
        for i, direction in enumerate(directions):
            neighbor_node = self.get_neighbor_node_id(node_id, direction)
            if neighbor_node:
                goal_distances[0, i] = self.get_distance(neighbor_node, goal_node_id)
            else:
                goal_distances[0, i] = np.inf

        return goal_distances


def generate_random_goals(env_grid, n_goals, row_limits=None, column_limits=None):
    """
    generate n random goals within the grid that are given by coordinated of unoccupied cells
    :return: list of n_goals
    """

    raise NotImplementedError

    if row_limits is None:
        row_limits = (0, env_grid.nrows)
    if column_limits is None:
        column_limits = (0, env_grid.ncols)

    generated_goals = list()

    if n_goals == "ALL":
        for coords in env_grid.all_coords:
            if env_grid[coords] == 0:
                generated_goals.append(coords)
        return generated_goals


    while (len(generated_goals) < n_goals):
        i = np.random.randint(row_limits[0], row_limits[1])
        j = np.random.randint(column_limits[0], column_limits[1])

        coods = (i, j)
        if coods not in generated_goals and env_grid.is_free(coods):
            generated_goals.append(coods)

    return generated_goals


def get_value_functions(env_grid, generated_goals, gamma):
    """
    generate 3D array that hold value functions for all n goals in generated goals list
    :param env_grid:
    :param generated_goals:
    :return: 3D grid that contains all value functions
    """

    n_goals = len(generated_goals)

    value_fcts = np.ones((env_grid.nrows, env_grid.ncols, n_goals)) * 0

    for g, goal in enumerate(generated_goals):

        print(f"at {g} of {n_goals}")

        goal = generated_goals[g]
        goal_node_id = env_grid.id_from_coords(goal)

        for coords in env_grid.all_coords:
            node_id = env_grid.id_from_coords(coords)
            value_fcts[coords][g] = env_grid.get_node_value(node_id, goal_node_id, gamma)

    return value_fcts


def get_policies(env_grid, generated_goals):
    """
    generate 3D array that holds policy for all n goals in generated goals list
    :param env_grid:
    :param generated_goals:
    :return: 3D grid that contains all value functions
    """

    n_goals = len(generated_goals)
    n_actions = len(PB_ACTIONS)
    policies = np.ones((env_grid.nrows, env_grid.ncols, n_goals, n_actions)) * np.nan

    for g, goal in enumerate(generated_goals):

        print(f"at {g} of {n_goals}")

        goal = generated_goals[g]
        goal_node_id = env_grid.id_from_coords(goal)

        node_id = 0
        for coords in env_grid.all_coords:
            nb_distances = env_grid.get_neighboring_distances(node_id, goal_node_id)
            policy = softmax(SOFTMAX_MULTIPLIER/(nb_distances+SOFTMAX_EPSILON))
            policies[coords][g][:] = policy
            node_id += 1

    return policies


def plot_optimal_actions(ax, rows, cols, goal, distances, offset, color):
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
            ax.text(j, i+offset, best_action_icons, color=color, ha='center', va='center')


def compute_weighted_average_policies(env_grid, policies, generated_goals):

    n_actions = len(PB_ACTIONS)
    avg_policies = np.zeros((env_grid.nrows, env_grid.ncols, 1, n_actions))

    for coords in env_grid.all_coords:
            curr_node_id = env_grid.id_from_coords(coords)

            state_policies = policies[coords][:, :]
            goal_distances = np.ones((1, len(generated_goals)))*np.nan

            for goal_id, goal in enumerate(generated_goals):
                goal_node_id = env_grid.id_from_coords(goal)
                goal_distances[0, goal_id] = env_grid.get_distance(curr_node_id, goal_node_id)

            # p(g|s) assumed to be proportional to GOAL_ASSIGNMENT_DECAY_FACTOR^(distance_to_g)
            decay_factor = GOAL_ASSIGNMENT_DECAY_FACTOR
            goal_probabilities = np.ones((1, len(generated_goals)))*decay_factor
            goal_probabilities = goal_probabilities**goal_distances

            for goal_id in range(len(generated_goals)):
                avg_policies[coords][0, :] += goal_probabilities[0, goal_id]*state_policies[goal_id, :]

            avg_policies[coords][0, :] = avg_policies[coords][0, :]/np.sum(avg_policies[coords][0, :])

    return avg_policies


def colorplot_values(ax, values):
    ax.imshow(values)


def plot_optimal_action_for_all_goals(ax, policies, init_offset):
    colors = ["yellow", "red", "blue", "green"]
    offset = copy.deepcopy(init_offset)
    for g in range(policies.shape[2]):
        plot_optimal_actions(ax, policies.shape[0], policies.shape[1], g, policies, offset, colors[g])
        offset += 0.15

    return offset


def plot_optimal_action_for_average_policy(ax, avg_policy, curr_offset):
    plot_optimal_actions(ax, avg_policy.shape[0], avg_policy.shape[1], 0, avg_policy, curr_offset, "black")


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


def make_plots(avg_values, policies, avg_policies, goals):
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
    env_grid = Grid(FNAME_PLAN)
    env_grid.convert_to_graph(diagonal_movements_possible=False)

    # generate or specify goals
    #generated_goals = generate_random_goals(env_grid, N_GOALS, ROW_LIMITS, COLUMN_LIMITS)
    horiz_center = int(env_grid.ncols/2)
    generated_goals = [(0, horiz_center), (2, 0), (2, env_grid.ncols-1), (5, horiz_center)]

    # compute value functions for all goals
    value_functions = get_value_functions(env_grid, generated_goals, GAMMA)


    # compute policies for all goals
    policies = get_policies(env_grid, generated_goals)
    avg_values = np.mean(value_functions, axis=2)

    # compute weighted average policy
    avg_policy = compute_weighted_average_policies(env_grid, policies, generated_goals)

    # make various plots
    make_plots(avg_values, policies, avg_policy, generated_goals)

if __name__ == "__main__":
    main()