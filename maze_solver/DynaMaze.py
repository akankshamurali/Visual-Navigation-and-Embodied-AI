from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load and convert the image to grayscale
image_path = Path('img/maze_1.jpeg').absolute().__str__()
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Step 1: Thresholding to get a binary image
# Invert colors for skeletonization
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# plt.imshow(binary_image)
# plt.show()

dilatation_size = 2
element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                    (dilatation_size, dilatation_size))
dilated_edges = cv2.dilate(binary_image, element)

# plt.imshow(dilated_edges)
# plt.show()

# Find contours and get bounding box of the largest contour (the maze boundary)
contours, _ = cv2.findContours(
    dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)
diff = 2
# Crop the image to the bounding box of the mazezx
cropped_image = dilated_edges[y+diff:y+h-diff, x+diff:x+w-diff]
cropped_image = cropped_image.astype(np.float64)
for i in range(0, cropped_image.shape[0]):
    for j in range(0, cropped_image.shape[1]):
        if cropped_image[i, j] == 255:
            cropped_image[i, j] = -1

ROWS = cropped_image.shape[0]
COLS = cropped_image.shape[1]
S = (7, 6)
G = (9, 253)
# BLOCKS = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]
BLOCKS = []
ACTIONS = ["left", "up", "right", "down"]


class Maze:

    def __init__(self):
        self.rows = ROWS
        self.cols = COLS
        self.start = S
        self.goal = G
        self.blocks = BLOCKS
        self.state = S
        self.end = False
        # init maze
        self.maze = np.zeros((self.rows, self.cols))
        for b in self.blocks:
            self.maze[b] = -1

    def nxtPosition(self, action):
        r, c = self.state
        if action == "left":
            c -= 1
        elif action == "right":
            c += 1
        elif action == "up":
            r -= 1
        else:
            r += 1

        if (r >= 0 and r <= self.rows-1) and (c >= 0 and c <= self.cols-1):
            if (r, c) not in self.blocks:
                self.state = (r, c)
        return self.state

    def giveReward(self):
        if self.state == self.goal:
            self.end = True
            return 1
        else:
            return 0

    def showMaze(self):
        self.maze[self.state] = 1
        for i in range(0, self.rows):
            print('-------------------------------------')
            out = '| '
            for j in range(0, self.cols):
                if self.maze[i, j] == 1:
                    token = '*'
                if self.maze[i, j] == -1:
                    token = 'z'
                if self.maze[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-------------------------------------')


class DynaAgent:

    def __init__(self, exp_rate=0.3, lr=0.1, n_steps=5, episodes=1):
        self.maze = Maze()
        self.state = S
        self.actions = ACTIONS
        self.state_actions = []  # state & action track
        self.exp_rate = exp_rate
        self.lr = lr

        self.steps = n_steps
        self.episodes = episodes  # number of episodes going to play
        self.steps_per_episode = []

        self.Q_values = {}
        # model function
        self.model = {}
        for row in range(ROWS):
            for col in range(COLS):
                self.Q_values[(row, col)] = {}
                for a in self.actions:
                    self.Q_values[(row, col)][a] = 0

    def chooseAction(self):
        # epsilon-greedy
        mx_nxt_reward = -999
        action = ""

        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
        else:
            # greedy action
            current_position = self.state
            # if all actions have same value, then select randomly
            if len(set(self.Q_values[current_position].values())) == 1:
                action = np.random.choice(self.actions)
            else:
                for a in self.actions:
                    nxt_reward = self.Q_values[current_position][a]
                    if nxt_reward >= mx_nxt_reward:
                        action = a
                        mx_nxt_reward = nxt_reward
        return action

    def reset(self):
        self.maze = Maze()
        self.state = S
        self.state_actions = []

    def play(self):
        self.steps_per_episode = []

        for ep in range(self.episodes):
            while not self.maze.end:

                action = self.chooseAction()
                self.state_actions.append((self.state, action))

                nxtState = self.maze.nxtPosition(action)
                reward = self.maze.giveReward()
                # update Q-value
                self.Q_values[self.state][action] += self.lr*(reward + np.max(
                    list(self.Q_values[nxtState].values())) - self.Q_values[self.state][action])

                # update model
                if self.state not in self.model.keys():
                    self.model[self.state] = {}
                self.model[self.state][action] = (reward, nxtState)
                self.state = nxtState

                # loop n times to randomly update Q-value
                for _ in range(self.steps):
                    # randomly choose an state
                    rand_idx = np.random.choice(range(len(self.model.keys())))
                    _state = list(self.model)[rand_idx]
                    # randomly choose an action
                    rand_idx = np.random.choice(
                        range(len(self.model[_state].keys())))
                    _action = list(self.model[_state])[rand_idx]

                    _reward, _nxtState = self.model[_state][_action]

                    self.Q_values[_state][_action] += self.lr*(_reward + np.max(
                        list(self.Q_values[_nxtState].values())) - self.Q_values[_state][_action])
            # end of game
            if ep % 10 == 0:
                print("episode", ep)
            self.steps_per_episode.append(len(self.state_actions))
            self.reset()


if __name__ == "__main__":
    N_EPISODES = 50
    # comparison
    agent = DynaAgent(n_steps=0, episodes=N_EPISODES)
    agent.play()

    steps_episode_0 = agent.steps_per_episode

    agent = DynaAgent(n_steps=5, episodes=N_EPISODES)
    agent.play()

    steps_episode_5 = agent.steps_per_episode

    agent = DynaAgent(n_steps=50, episodes=N_EPISODES)
    agent.play()

    steps_episode_50 = agent.steps_per_episode

    plt.figure(figsize=[10, 6])

    plt.ylim(0, 900)
    plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
    plt.plot(range(N_EPISODES), steps_episode_5, label="step=5")
    plt.plot(range(N_EPISODES), steps_episode_50, label="step=50")

    plt.legend()
