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
        # state required for physics and renderer
        # [x, y, theta, u1, u2, omega]
        self.physics_state = np.array([x, y, 0, 0, 0, 0])
        self.prop_strength = prop_strength
        self.mass = mass
        self.r = r
        self.mI = ((r * 2) ** 2) * mass * 1 / 12
        self.time_alive = 0
        self.hid = hid
        self.keyboard = Controller()
        self.current_input = self.Input.NEITHER
        self.current_action = np.array(
            [self.Input.LEFT, self.Input.RIGHT, self.Input.BOTH, self.Input.NEITHER]
        )
        self.dt = 0.05
        # state passed to agent for training
        # [vx, vy, acc_x, acc_y, acc_ang, time]
        self.agent_state = np.array([0, 0, 0, 0, 0, 0])

    def reset(self, x=10, y=10):
        self.physics_state = np.array([x, y, 0, 0, 0, 0])

    def get_agent_state(self):
        # ich bein und Javaer
        return self.agent_state

    def dynamics(self, u1, u2, theta):
        acc_x = -(u1 + u2) * math.sin(theta) * 1 / self.mass
        acc_y = (u1 + u2) * math.cos(theta) * 1 / self.mass - scipy.constants.g
        acc_ang = self.r * (u1 - u2) * 1 / self.mI
        return (acc_x, acc_y, acc_ang)

    def update(self):
        self.input()
        x, y, theta, vx, vy, omega = self.physics_state
        acc = self.dynamics(vx, vy, theta)

        vx += acc[0] * self.dt
        vy += acc[1] * self.dt
        omega += acc[2] * self.dt

        time = self.agent_state[-1]

        x += vx * self.dt
        y += vy * self.dt
        theta += math.radians(omega * self.dt)

        self.agent_state = np.array([vx, vy, acc[0], acc[1], acc[2], time + 1])
        self.physics_state = np.array([x, y, theta, vx, vy, omega])

    def edges(self):
        x, y, theta = self.physics_state[0:3]
        x1 = x - self.r * math.cos(theta)
        y1 = y - self.r * math.sin(theta)
        x2 = x + self.r * math.cos(theta)
        y2 = y + self.r * math.sin(theta)

        return ([x1, x2], [y1, y2])

    def crash(self):
        # ich bein un haskller
        return any(map(lambda x: x < 0, sum(self.edges(), [])))

    def input(self):
        match self.current_input:
            case self.Input.LEFT:
                self.physics_state[3] += self.prop_strength
            case self.Input.RIGHT:
                self.physics_state[4] += self.prop_strength
            case self.Input.BOTH:
                self.physics_state[3] += self.prop_strength
                self.physics_state[4] += self.prop_strength
            case self.Input.NEITHER:
                pass

    def set_input(self, instruction):
        self.current_input = instruction

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
            # print(self.current_input)

    def human_input(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
