import scipy.constants
import scipy.integrate
import math
import numpy as np
import numpy.typing as npt


class Quad2d:
    t: float
    dt: float
    y0: list[float]
    y: list[float] | npt.NDArray[np.float64]
    input: list[float]
    m: float
    l: float
    j: float

    def __init__(self) -> None:
        self.t = 0.0
        self.dt = 0.1
        # x pos, y pos, theta, vx, vy, omega
        # i.e. f(t, y, dy/dt)
        self.y0 = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        self.y = self.y0
        # F1, F2
        self.input = [0.0, 5.0]
        self.m = 0.18  # Kg
        self.l = 0.086  # m
        self.j = 2.5e-4  # Kgm^2 moment of inertia

    def u(self) -> list[float]:
        return [sum(self.input), (self.l / 2) * (self.input[0] - self.input[1])]

    def reset(self) -> None:
        self.y = self.y0

    def dynamics(self, t: float, y: list[float] | npt.NDArray[np.float64]) -> list[float]:
        """
        given the current state y, and u1(t) and u2(t), return the second derivatives of state
        i.e. dy/dt
        """
        _, _, theta, vx, vy, omega = y
        acc_x = -self.u()[0] * math.sin(theta) * 1 / self.m
        acc_y = -scipy.constants.g + (self.u()[0] * math.cos(theta) * 1 / self.m)
        alpha = self.u()[1] / self.j
        return [vx, vy, omega, acc_x, acc_y, alpha]

    def solve(self) -> npt.NDArray[np.float64]:
        t_span: tuple[float, float] = (self.t, self.t + self.dt)
        self.t += self.dt
        solution = scipy.integrate.solve_ivp(
            self.dynamics, t_span, self.y, method="RK45"
        )
        return solution.y[:, -1]

    def update(self) -> None:
        self.y = self.solve()

    def edges(self) -> tuple[list[float], list[float]]:
        x, y, theta, _, _, _ = self.y
        x1 = x - self.l * math.cos(theta)
        y1 = y - self.l * math.sin(theta)
        x2 = x + self.l * math.cos(theta)
        y2 = y + self.l * math.sin(theta)

        return ([x1, x2], [y1, y2])

    def crash(self) -> bool:
        # ich bein un haskller
        return any(map(lambda x: x < 0, sum(self.edges(), [])))

    def set_input(self, f1: float, f2: float) -> None:
        self.input = [f1, f2]
