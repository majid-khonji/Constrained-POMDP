import numpy as np
from itertools import product


class Instance:  # compatible with durative CC-POMDP
    name = ""  # instance name
    states = []
    observations = []
    actions = []
    horizon = 3

    b0 = {}  # initial belief, use dict for non-zero values only
    goal_state = 0
    states_reachable_to = {} # states that can reach s [to speedup prior computation]

    delta = 0
    risk_states = [] # states such that risk_model(s,a) > 0 for any a


    def trans_model(self, s, a):
        return {}

    def obs_model(self, s, a):
        return {}

    def reward_model(self, s, a):
        return {}

    def risk_model(self, s, a):
        return {}

    def duration_model(self,q):
        return np.floor((len(q)+1)/2)

    def reward_heuristic(self, s, a):
        pass
    def risk_heuristic(self,s,a):
        pass

    # helpers
    def action_to_string(self, a):
        return ""

    def print_state(self):  # user friendly state message
        pass


class GridInstance(Instance):
    name = "Grid"
    grid_size = (4, 4)
    occupancy = None
    actions = [0, 1, 2, 3]  # left, up, right, down [clockwise]
    observations = [0, 1, 2]  # number of adjacent walls
    risk_states = None
    u_heuristic_grid = None

    # TODO allow walls within grid

    def __init__(self, size=(5, 5), risk_idx=[(0,0), (3, 0), (3, 1),(1,3),(1,4)], start_s=(4, 0), goal_s=(0, 4), delta = .1, u_h = "manhattan"):
        self.grid_size = size
        self.start_state = start_s
        self.b0 = {start_s: 1}
        self.goal_state = goal_s
        self.occupancy = np.zeros(size)
        for (i, j) in risk_idx:
            self.occupancy[i, j] = 1
        self.states = [(i, j) for i, j in product(range(size[0]), range(size[1]))]
        self.risk_states = risk_idx
        self.delta = delta


        self.u_heuristic_grid = np.zeros(size)
        for i in np.arange(size[0]):
            for j in np.arange(size[1]):
                if u_h == "manhattan":
                    self.u_heuristic_grid[i][j] = np.abs(i - self.goal_state[0]) + np.abs(j - self.goal_state[1])
                else:
                    self.u_heuristic_grid[i][j] =  np.sqrt((i - self.goal_state[0]) ** 2 + (j - self.goal_state[1]) ** 2)

        for s in self.states:
            self._update_states_reachable_to(s)


    def trans_model(self, s, a):
        # 85% same direction, 0.075% in either ways
        i = s[0]
        j = s[1]
        if a == 0 or a == 2:  # horizontal
            if a == 0:  # left
                same = (i, j - 1 if j > 0 else 0)
            if a == 2:  # right
                same = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)
            left = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
            right = (i - 1 if i > 0 else 0, j)
        if a == 1 or a == 3:  # vertical
            if a == 1:  # up
                same = (i - 1 if i > 0 else 0, j)
            if a == 3:  # down
                same = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
            left = (i, j - 1 if j > 0 else 0)
            right = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)

        if same == right:
            next = {same: .85 + 0.075, left: .075}
        elif same == left:
            next = {same: .85 + 0.075, right: .075}
        else:
            next = {same: .85, left: .075, right: .075}

        return next

    # returns number of walls
    def obs_model(self, s, a):
        (i, j) = s
        if s == (self.grid_size[0] - 1, self.grid_size[1] - 1) or s == (0, 0):
            return {2: .85, 1: 0.075, 0: 0.075}
        elif i == 0 or i == self.grid_size[0] - 1 or j == 0 or j == self.grid_size[1] - 1:
            return {2: 0.075, 1: .85, 0: 0.075}
        else:
            return {2: 0.075, 1: 0.075, 0: .85}

    def reward_model(self, s, a):
        if s == self.goal_state: # or s in self.risk_states:
            return 0
        return 1
        # (i,j) = s
        #
        # if a == 0: #left
        #     next = (i,j-1 if j > 0 else 0)
        # elif a == 1: # up
        #     next = (i-1 if i > 0 else 0, j)
        # elif a == 2: # right
        #     next = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)
        # elif a == 3:  # down
        #     next = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
        # if next == self.goal_state:
        #     return 0
        # return 1

    def reward_heuristic(self, s, a):
        # return np.sqrt((s[0] - self.goal_state[0])**2 + (s[1] - self.goal_state[1])**2)
        return self.u_heuristic_grid[s[0]][s[1]]

        # (i,j) = s

        # if a == 0: #left
        #     next = (i,j-1 if j > 0 else 0)
        # elif a == 1: # up
        #     next = (i-1 if i > 0 else 0, j)
        # elif a == 2: # right
        #     next = (i, j + 1 if j < self.grid_size[1] - 1 else self.grid_size[1] - 1)
        # elif a == 3:  # down
        #     next = (i + 1 if i < self.grid_size[0] - 1 else self.grid_size[0] - 1, j)
        #
        # return np.abs(next[0] - self.goal_state[0]) + np.abs(next[1] - self.goal_state[1])

    def risk_model(self, s, a=None):
        return self.occupancy[s[0]][s[1]]


    def _update_states_reachable_to(self, s):
        (i, j) = s

        self.states_reachable_to[s] = [(i - 1, j) if i > 0 else (i, j),
                                       (i + 1, j) if i < self.grid_size[0] - 1 else (i, j),
                                       (i, j - 1) if j > 0 else (i, j),
                                       (i, j + 1) if j < self.grid_size[1] - 1 else (i, j)]
        if i == 0 or i == self.grid_size[0] - 1 or j == 0 or j == self.grid_size[1]:
            self.states_reachable_to[s].append(s)

        self.states_reachable_to[s] = set(self.states_reachable_to[s])


    def action_to_string(self, a):
        if a == 0:
            return "←"
        if a == 1:
            return "↑"
        if a == 2:
            return "→"
        if a == 3:
            return "↓"

    def print_state(self):
        print("┌", end='')
        for j in np.arange(self.grid_size[1]):
            print("──", end='')
        print("─┐")
        for i in np.arange(self.grid_size[0]):
            print("│ ", end='')
            for j in np.arange(self.grid_size[1]):
                if (i, j) == self.start_state:
                    print("S ", end='')
                elif (i, j) == self.goal_state:
                    print("G ", end='')
                elif self.occupancy[i, j] == 0:
                    print(". ", end='')
                elif self.occupancy[i, j] == 1:
                    print("🔥", end='')  # prints fire, but not visible here
            print("│")  # new line
        print("└", end='')
        for j in np.arange(self.grid_size[1]):
            print("──", end='')
        print("─┘")
