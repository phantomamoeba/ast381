#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #3
#March 12, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""

NEGATIVE_V = False

import numpy as np
import matplotlib.pyplot as plt

p_max = 1.0 #arbitrary scale
v_max = 1.0 #arbitrary scale

x_step = 0.01
x_grid = np.arange(-15. *v_max, 15.* v_max + x_step, x_step)

#uncomment to make negative
if NEGATIVE_V:
    v_max = -1.0


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
        return 0.5 * (1. - x/(v_max*t))*p_max


def make_plot(y_0,t,label=None):
    """
    Make a plot at a particular time. Given the analytic solution, there is no need to evolve this.
    :param t:
    :return:
    """

    y = np.zeros(y_0.shape)
    for i in range(len(x_grid)):
        y[i] = p(x_grid[i],t)

    # todo: add decoration (title, etc)
    plt.plot(x_grid,y,label=label)


def main():

    y = ic()

    plt.close('all')

    make_plot(y, 0,label="t=0")
    make_plot(y, 1,label="t=1")
    make_plot(y, 5,label="t=5")
    make_plot(y, 10,label="t=10")



    if NEGATIVE_V:
        plt.title("Negative Velocity")
        plt.legend()
        plt.savefig("/home/dustin/code/python/ast381_compastro/hw3/negative.png")
    else:
        make_plot(y, 50, label="t=50") #add one more to show approach horizontal
        make_plot(y, 100, label="t=100")  # add one more to show approach horizontal
        plt.legend()
        plt.title("Positive Velocity")
        plt.savefig("/home/dustin/code/python/ast381_compastro/hw3/positive.png")




if __name__ == '__main__':
    main()

