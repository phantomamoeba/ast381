#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #3
#March 12, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""

import numpy as np
import matplotlib.pyplot as plt

p_max = 1.0 #arbitrary scale
v_max = 1.0 #arbitrary scale

x_step = 0.01
x_grid = np.arange(-10. *v_max, 10 * v_max + x_step, x_step)



def ic():
    """
    Initial conditions
    :return:
    """
    y = np.zeros(x_grid.shape)
    y[np.where(x_grid < 0)] = p_max

    return y


def p(x,t):
    """
    Solution per eq (4)
    :param x:
    :param t:
    :return:
    """

    if x <= (-1 * v_max * t):
        return p_max
    elif x >= (v_max * t):
        return 0.0
    else:
        return 0.5 * (1. - x/(v_max*t))


def make_plot(y_0,t):
    """
    Make a plot at a particular time. Given the analytic solution, there is no need to evolve this.
    :param t:
    :return:
    """

    y = np.zeros(y_0.shape)
    for i in range(len(x_grid)):
        y[i] = p(x_grid[i],t)

    # todo: add decoration (title, etc)
    plt.plot(x_grid,y)


