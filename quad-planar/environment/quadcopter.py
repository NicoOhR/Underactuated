import scipy.constants
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from pynput import keyboard
from pynput.keyboard import Controller


class Quadcopter2d:

    class Input(Enum):
        LEFT = 1
        RIGHT = 2
        BOTH = 3
        NEITHER = 4

    def __init__(self, mass=5, r=5, x=10, y=10, prop_strength=6, hid: bool = False):
        self.prop_strength = prop_strength
        self.mass = mass
        self.r = r
        self.mI = ((r * 2) ** 2) * mass * 1 / 12
        self.state = np.array([x, y, 0, 0, 0, 0])  # [x, y, theta, u1, u2, omega]
        self.hid = hid
        self.keyboard = Controller()
        self.current_input = self.Input.NEITHER

    def dynamics(self, u1, u2, theta):
        acc_x = -(u1 + u2) * math.sin(theta) * 1 / self.mass
        acc_y = (u1 + u2) * math.cos(theta) * 1 / self.mass - scipy.constants.g
        acc_ang = self.r * (u1 - u2) * 1 / self.mI
        return (acc_x, acc_y, acc_ang)

    def update(self, u1, u2, dt):
        self.input()
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

    def input(self):
        match self.current_input:
            case self.Input.LEFT:
                self.state[3] += self.prop_strength
            case self.Input.RIGHT:
                self.state[4] += self.prop_strength
            case self.Input.BOTH:
                self.state[3] += self.prop_strength
                self.state[4] += self.prop_strength
            case self.Input.NEITHER:
                pass

    def on_press(self, key):
        if self.hid is True:
            if key.char == "p":
                self.current_input = self.Input.BOTH
            elif key.char == "u":
                self.current_input = self.Input.LEFT
            elif key.char == "o":
                self.current_input = self.Input.RIGHT
            else:
                self.current_input = self.Input.NEITHER
            print(self.current_input)

    def human_input(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
