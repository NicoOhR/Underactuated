from .quadcopter import Quadcopter2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

import matplotlib.pyplot as plt
import time


class QuadRender:
    def __init__(self, quad):
        self.dt = 0.0001
        self.quad = quad
        self.initialize_plot()

    def initialize_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        (self.quadcopter_body,) = self.ax.plot([], [], "o-", lw=2, label="Quadcopter")
        (self.trail,) = self.ax.plot([], [], "r--", lw=1, label="Trail")
        self.time_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)
        self.ax.set_xlim(-50, 200)
        self.ax.set_ylim(0, 100)
        self.trajectory_x = []
        self.trajectory_y = []
        self.quadcopter_body.set_data([], [])
        self.trail.set_data([], [])
        self.time_text.set_text("")

    def render(self, frame):
        edges = self.quad.edges()
        state = self.quad.physics_state
        e1, e2 = edges
        self.quadcopter_body.set_data(e1, e2)
        self.trajectory_x.append(state[0])
        self.trajectory_y.append(state[1])
        # self.trail.set_data(self.trajectory_x, self.trajectory_y)
        self.time_text.set_text(f"Time: {frame * self.dt:.1f}s")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


if __name__ == "__main__":
    quad = Quadcopter2d(hid=True)
    quad_anim = QuadRender(quad)
    frame = 0
    try:
        while True:
            quad.update()
            quad.human_input()
            quad_anim.render(frame)
            frame += 1
            time.sleep(quad_anim.dt)
    except KeyboardInterrupt:
        print("Animation stopped.")
        plt.ioff()
        plt.show()
