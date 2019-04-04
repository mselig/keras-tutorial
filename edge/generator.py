# -*- coding: utf-8 -*-
"""
    hosts the class `Generator`
"""

import numpy  as np
import pandas as pd


Theta = lambda x: np.heaviside(x, 0)
__residual = lambda x, y, a, b: a * x + b - y
__f = lambda x, y, a, b: Theta(__residual(x, y, a, b)) * __residual(x, y, a, b) ** 2 / (-2 * a)
integral = lambda x1, x2, y1, y2, a, b: __f(x2, y2, a, b) - __f(x2, y1, a, b) - __f(x1, y2, a, b) + __f(x1, y1, a, b)


class Generator(object):
    """
        implements a data generator for images with an edge
    """
    def __init__(self, resolution=7, seed=42):
        """
            initialises a `Generator` with an image resolution and random seed
        """
        self.dim = max(2, int(resolution))
        np.random.seed(seed)

        self.shuffle = np.arange(self.dim ** 2)
        np.random.shuffle(self.shuffle)  ## in place

        self.header = []
        for index in range(self.dim):
            for jndex in range(self.dim):
                self.header += [f"x_{index}{jndex}"]
        for index in range(self.dim):
            for jndex in range(self.dim):
                self.header += [f"z_{index}{jndex}"]
        self.header += ["y_horizontal"]
        self.header += ["y_1-hot-slope_lt-1"]
        self.header += ["y_1-hot-slope_lt_0"]
        self.header += ["y_1-hot-slope_lt_1"]
        self.header += ["y_1-hot-slope_gt_1"]
        self.header += ["y_angle"]
        self.header += ["y_slope"]
        self.header += ["y_intercept"]
        self.header += ["y_root"]
        self.header += ["y_Ax"]
        self.header += ["y_Ay"]
        self.header += ["y_Bx"]
        self.header += ["y_By"]
        self.header += ["y_length"]
        self.header += ["y_area"]
        self.header += ["y_noise"]

    def generate_data(self, amount=100, bias=0.4, extra_bias=False):
        """
            generates a data frame containg an amount of images
        """
        num = int(amount)
        assert num > 0
        bias = np.clip(bias, 0, 1)
        alternate = bool(extra_bias)

        data = np.empty((num, len(self.header)))
        for index in range(num):
            data[index] = self.__generate_sample(bias, alternate)

        return pd.DataFrame(data, columns=self.header)

    def __generate_sample(self, bias, alternate):
        """
            generates an image with an edge and all meta data
        """
        Ax, Ay, Bx, By, angle, slope, intercept, root, length, area = self.__generate_edge(bias, alternate)
        Y_slopes = self.__encode_slopes(slope)
        x_positions, y_positions = np.meshgrid(np.arange(self.dim + 1) / self.dim, \
                                               np.arange(self.dim + 1) / self.dim)
        X = np.empty((self.dim, self.dim))
        for index in range(self.dim):
            for jndex in range(self.dim):
                X[index, jndex] = integral( \
                        x_positions[index, jndex], x_positions[index + 1, jndex + 1], \
                        y_positions[index, jndex], y_positions[index + 1, jndex + 1], \
                        slope, intercept) * self.dim ** 2
        noise = np.random.lognormal(mean=0.7, sigma=0.5) ## ~1..4% noise
        X = np.clip(X + np.random.normal(scale=0.01 * noise, size=X.shape), 0, 1).flatten()
        Y = np.concatenate((Y_slopes, [angle, slope, intercept, root, Ax, Ay, Bx, By, length, area, noise]))
        Z = X[self.shuffle]
        return np.round(np.concatenate((X, Z, Y)), 5)

    def __generate_edge(self, bias, alternate):
        """
            generates an edge
        """
        Ox, Oy = np.random.random(size=2) * (1 - bias) + bias / 2
        if(alternate):
            Px, Py = np.random.random(size=2) * (1 - bias) + bias / 2
            if(Ox > Px):
                Ox, Oy, Px, Py = Px, Py, Ox, Oy ## A fore B
            slope = (Py - Oy) / (Px - Ox)
            angle = np.arctan(slope) ## [rad]
        else:
            angle = (np.random.random() - 0.5) * np.pi ## [rad]
            slope = np.tan(angle) ## y = slope * x + intercept
        intercept = Oy - slope * Ox
        root = 0 - intercept / slope
        Ay = np.clip(intercept, 0, 1)
        Ax = (Ay - intercept) / slope
        By = np.clip(slope + intercept, 0, 1)
        Bx = (By - intercept) / slope
        length = np.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2) ## Pytagoras
        area = integral(0, 1, 0, 1, slope, intercept) ## == $\int_0^1 \mathrm{d}x \int_0^1 \mathrm{d}y Theta(a * x + b - y)$
        return (Ax, Ay, Bx, By, angle, slope, intercept, root, length, area)

    def __encode_slopes(self, slope):
        """
            binary- and 1=hot-encodes different slopes
        """
        hot = np.empty(5, dtype=np.float64)
        hot[0] = (-1 < slope <  1) ## horizontal
        hot[1] = (     slope < -1) ## strong decrease
        hot[2] = (-1 < slope <  0) ##   weak decrease
        hot[3] = ( 0 < slope <  1) ##   weak increase
        hot[4] = ( 1 < slope     ) ## strong increase
        return hot


#if __name__ == "__main__":
#    data = Generator().generate_data(1000)

