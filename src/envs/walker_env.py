import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class SimpleWalkerEnv(gym.Env):
    """
    A very lightweight continuous-control walker-style environment.
    State: [position, velocity]
    Action: force applied to walker
    Reward: forward progress - energy penalty
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.viewer = None
        self.fig = None
        self.ax = None
        self.walker_plot = None
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.t = 0
        return np.array([self.pos, self.vel], dtype=np.float32), {}

    def step(self, action):
        force = float(action.squeeze())
        self.vel += 0.1 * force
        self.pos += self.vel
        self.t += 1

        reward = self.vel - 0.01 * abs(force)
        terminated = False
        truncated = self.t >= 200

        obs = np.array([self.pos, self.vel], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if self.viewer is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 3))
            self.ax.set_xlim(-10, 250)
            self.ax.set_ylim(-1, 1)
            self.ax.axhline(0, color='black', lw=2)
            self.walker_plot, = self.ax.plot([], [], 'o', markersize=20, color='blue')
            self.viewer = self.fig
            plt.title("Simple Walker")
            plt.xlabel("Position")
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        self.walker_plot.set_data([self.pos], [0.1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None
            plt.ioff()
