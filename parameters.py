#FNAME_PLAN = "plan_3.txt"
#FNAME_PLAN = "maze/new_maze.txt"
FNAME_PLAN = "plan_free_large.txt"
GAMMA = 0.8
N_GOALS = "ALL"
NODE_DISTANCE_IF_NO_PATH = 1000 # distance returned if no path exists
SOFTMAX_MULTIPLIER = 1000 # multiplier used for softmax policy computation (similar to policy temperature; the higher the more like a one-hot policy)
SOFTMAX_EPSILON = 0.0001 # epsilon used to make softmax numerically stable
GOAL_ASSIGNMENT_DECAY_FACTOR = 0.6


#ROW_LIMITS = (0, 5)
#COLUMN_LIMITS = (6, 11)

ROW_LIMITS = None
COLUMN_LIMITS = None