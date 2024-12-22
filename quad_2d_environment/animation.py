from quadcopter import Quadcopter2d
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation


class QuadAnimation:

    def __init__(self):
        self.dt = 0.05

        self.fig, self.ax = plt.subplots()

        self.quadcopter_body, = self.ax.plot([], [], 'o-', lw=2, label="Quadcopter")
        self.trail, = self.ax.plot([], [], 'r--', lw=1, label="Trail")
        self.time_template = "Time: {:.1f}s"
        self.time_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.ax.set_xlim(-50, 50) 
        self.ax.set_ylim(0, 100) 

        self.trajectory_x = []
        self.trajectory_y = []
        self.quadcopter_body.set_data([], [])
        self.trail.set_data([], [])
        self.time_text.set_text('')
        self.quad = Quadcopter2d()

    def init_animation(self):
        self.quadcopter_body.set_data([], [])
        self.trail.set_data([], [])
        self.time_text.set_text('')
        return self.quadcopter_body, self.trail, self.time_text

    def update(self, frame):

        (e1, e2) = self.quad.edges() 
        self.quadcopter_body.set_data(e1, e2)

        self.trajectory_x.append(self.quad.state[0])
        self.trajectory_y.append(self.quad.state[1])
        self.trail.set_data(self.trajectory_x, self.trajectory_y)
        self.time_text.set_text(self.time_template.format(frame * self.dt))

        u1,u2 = self.quad.state[3:5]
        self.quad.update(u1, u2, self.dt)
        self.quad.crash()
        return self.quadcopter_body, self.trail, self.time_text

    def run(self):
        ani = animation.FuncAnimation(self.fig,self.update, frames=500, init_func=self.init_animation, blit=True, interval=self.dt*1000)
        plt.show()
        plt.close(self.fig)

if __name__ == "__main__":
    quad_anim = QuadAnimation()
    quad_anim.run()

