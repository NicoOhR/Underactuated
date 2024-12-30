import scipy.constants
import scipy.integrate
import math


class Quad2d:
    def __init__(self):
        self.t = 0.0
        self.dt = 0.1
        # x pos, y pos, theta, vx, vy, omega
        # i.e. f(t, y, dy/dt)
        self.y = [10.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        # F1, F2
        self.input = [0.0, 5.0]
        self.m = 0.18  # Kg
        self.l = 0.086  # m
        self.j = 2.5e-4  # Kgm^2 moment of inertia

        self.u1 = sum(self.input)
        self.u2 = self.l / 2 * (self.input[0] - self.input[1])

    def dynamics(self, t, y):
        """
        given the current state y, and u1(t) and u2(t), return the second derivatives of state
        i.e. dy/dt
        """
        _, _, theta, vx, vy, omega = y
        acc_x = -self.u1 * math.sin(theta) * 1 / self.m
        acc_y = -scipy.constants.g + (self.u1 * math.cos(theta) * 1 / self.m)
        alpha = self.u2 / self.j
        return [vx, vy, omega, acc_x, acc_y, alpha]

    def solve(self):
        t_span = (self.t, self.t + self.dt)
        self.t += self.dt
        solution = scipy.integrate.solve_ivp(
            self.dynamics, t_span, self.y, method="RK45"
        )
        print(solution.y[:, -1])
        return solution.y[:, -1]

    def update(self):
        self.y = self.solve()

    def edges(self):
        x, y, theta, _, _, _ = self.y
        print(self.y[0])
        x1 = x - self.l * math.cos(theta)
        y1 = y - self.l * math.sin(theta)
        x2 = x + self.l * math.cos(theta)
        y2 = y + self.l * math.sin(theta)

        return ([x1, x2], [y1, y2])
