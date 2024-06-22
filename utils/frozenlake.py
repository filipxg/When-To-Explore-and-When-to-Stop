import gym
from gym import spaces
import numpy as np
import torch


class FrozenLakeEnv(gym.Env):
    def __init__(self, grid):
        super(FrozenLakeEnv, self).__init__()
        self.grid = grid
        self.grid_height = len(grid)
        self.grid_width = len(grid[0])
        [self.start_y], [self.start_x] = np.where(np.array(grid) == 'S')
        self.slip_probability = 1 / 3  # Probability to slip to one side (so the chance of slipping in any direction is 2 times this value)
        assert self.slip_probability <= 1 / 2
        self.action_space = spaces.Discrete(4)  # Left, Down, Right, Up
        self.observation_space = spaces.Discrete(self.grid_height * self.grid_width)

        self.state_action_count = {}
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                self.state_action_count[(x, y)] = {0: 0, 1: 0, 2: 0, 3: 0}

    def reset(self):
        # Top left corner is 0, 0
        self.state = (self.start_x, self.start_y)
        return self.to_observation(self.state)

    def step(self, action):
        self.state_action_count[self.state][action] += 1
        x, y = self.state

        # Define possible actions for each chosen direction
        # Make sure the first action in the array is the action itself (no slip)
        possible_actions = {
            0: [0, 3, 1],  # Left
            1: [1, 0, 2],  # Down
            2: [2, 1, 3],  # Right
            3: [3, 2, 0]  # Up
        }

        # Choose a random action from the possible actions according to self.slip_probability
        p = self.slip_probability
        action = np.random.choice(possible_actions[action], p=[1 - 2 * p, p, p])
        # print("Actual action", ["left", "down", "right", "up"][action])

        # Move in the chosen direction if its within bounds
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.grid_height - 1:
            y += 1
        elif action == 2 and x < self.grid_width - 1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1

        self.state = (x, y)
        reward = 0
        done = False

        # Check state of the cell
        if self.grid[y][x] == 'X':
            reward = -5
            done = True
        elif self.grid[y][x] == 'G':
            reward = 10
            done = True

        return self.to_observation(self.state), reward, done, {}

    def get_aleatoric_uncertainty(self):
        # Define possible actions for each chosen direction
        # Make sure the first action in the array is the action itself (no slip)
        possible_actions = {
            0: [0, 3, 1],  # Left
            1: [1, 0, 2],  # Down
            2: [2, 1, 3],  # Right
            3: [3, 2, 0]  # Up
        }
        variance_map = []

        # Choose a random action from the possible actions according to self.slip_probability
        p = self.slip_probability

        for state in range(self.grid_width * self.grid_height):
            # Move in the chosen direction if its within bounds
            variances = {0: [], 1: [], 2: [], 3: []}
            # print(state)
            for wanted_action in possible_actions:
                for action in possible_actions[wanted_action]:
                    x, y = state % self.grid_width, state // self.grid_width
                    # print(x,y)
                    if action == 0 and x > 0:
                        x -= 1
                    elif action == 1 and y < self.grid_height - 1:
                        y += 1
                    elif action == 2 and x < self.grid_width - 1:
                        x += 1
                    elif action == 3 and y > 0:
                        y -= 1
                    # Check state of the cell
                    if self.grid[y][x] == 'X':
                        reward = -5
                    elif self.grid[y][x] == 'G':
                        reward = 10
                    else:
                        reward = 0
                    variances[wanted_action].append(reward)
            # print(variances)

            for k in variances:
                variances[k] = torch.tensor(np.var(variances[k]))
            variances = torch.stack(tuple(variances[k] for k in variances))
            variance_map.append(variances)

        return variance_map

    def to_observation(self, state):
        x, y = state
        return y * self.grid_width + x

    def render(self):
        grid = np.full((self.grid_height, self.grid_width), ' ')
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                grid[y, x] = self.grid[y][x]
        x, y = self.state
        grid[y, x] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()


def make_frozenlake(grid):
    return FrozenLakeEnv(grid)
