import scipy.constants
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from pynput import keyboard
from pynput.keyboard import Controller


class Quadcopter2d:
    def __init__(self, mass=5, r=5):
        self.mass = mass
        self.r = r
        self.mI = ((r * 2) ** 2) * mass * 1 / 12
        self.state = np.array([10, 10, 0, 0, 0, 0])  # [x, y, theta, u1, u2, omega]
        self.keyboard = Controller()

    def dynamics(self, u1, u2, theta):
        acc_x = -(u1 + u2) * math.sin(theta) * 1 / self.mass
        acc_y = (u1 + u2) * math.cos(theta) * 1 / self.mass - scipy.constants.g
        acc_ang = self.r * (u1 - u2) * 1 / self.mI
        return (acc_x, acc_y, acc_ang)

    def update(self, u1, u2, dt):
        x, y, theta, vx, vy, omega = self.state
        acc = self.dynamics(u1, u2, theta)

        vx += acc[0] * dt
        vy += acc[1] * dt
        omega += acc[2] * dt

        x += vx * dt
        y += vy * dt
        theta += omega * dt

        self.state = np.array([x, y, theta, vx, vy, omega])

    def edges(self):
        x, y, theta = self.state[0:3]
        x1 = x - self.r * math.cos(theta)
        y1 = y - self.r * math.sin(theta)
        x2 = x + self.r * math.cos(theta)
        y2 = y + self.r * math.sin(theta)

        # Floating point precision causes +/- 1e-7 of error
        # I simply cannot escape IEEE 754
        d = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        return ([x1, x2], [y1, y2])

    def crash(self):
        # ich bein un haskller
        return any(map(lambda x: x < 0, sum(self.edges(), [])))

    def input(self, u1, u2):
        self.state[3] = max(self.state[3] + 3, u1) if self.state[3] - u1 >= 3 else u1
        self.state[4] = max(self.state[4] + 3, u2) if self.state[4] - u2 >= 3 else u2

    def on_press(self, key):
        if key.char == "u":
            print("left")
            self.input(self.state[3] + 3, self.state[4])
        if key.char == "i":
            print("right")
            self.input(self.state[3], self.state[4] + 3)

    def human_input(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
