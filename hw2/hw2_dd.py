#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #2
#Februrary 26, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""

import numpy as np
import matplotlib.pyplot as plt

#need to think about space (x,y) and time (all x,y at a particular time step)
#maybe a 3 DIM ndarry x,y,t ?
#maybe not ... seems all are at t -> t+1 and need only space n-1 to n+1

dx = 2.*np.pi/1000.
dt = None
L = 10. * 2. * np.pi #10 cycles
cs = 1.0 #sound speed


def initial_wave(lam):

    x = np.array(np.arange(0.0,L+dx,dx))
    k = 2*np.pi/lam

    x0_idx = (np.abs(x - L/2.0)).argmin()
    x0 = x[x0_idx]

    y = np.cos(k*(x-x0))*np.exp((-1.*(x-x0)**2.)/2.)

    return x,y


def perfect_shift(y):
    """
    Sift perfectly, no calculations of y; just y(n) -> y((n+1)%L)
    (basically a rotate-right)
    """
    return np.append([y[-1]],y[0:-1])


def ftcs(_x,_y,_cs,_dx,_dt):
    """
    Advance 1 step using Forward-time, Centered-space
    :return:

    """


    #todo: try a single call with n-1 and (n+1)%nLen
    #n-1 will go negative at n == 0 and -1 is the last element so that's okay
    #(n+1) %nLen almost okay ... need to go from [n] to [0]  and maybe that works

    s = _cs * _dt / (2. * _dx)
    #in a loop ... think about a more pythonic way to do this

    next_y = np.zeros(np.shape(_y)) #next time step
    nLen = len(_y)
    for n in range(nLen):
        n_next = (n + 1) if n < nLen else 0
        n_prev = (n - 1) if n > 0 else nLen-1
        next_y[n] = _y[n] - s*(_y[n_next] - _y[n_prev])

    return next_y



def lax(_x,_y,_cs,_dx,_dt):
    """
    Advance 1 step using Lax
    :return:

    """

    s = _cs * _dt / (2. * _dx)
    # in a loop ... think about a more pythonic way to do this

    next_y = np.zeros(np.shape(_y))  # next time step
    nLen = len(_y)
    for n in range(nLen):
        n_next = (n + 1) if n < nLen else 0
        n_prev = (n - 1) if n > 0 else nLen - 1
        next_y[n] = 0.5 * (_y[n_next] + _y[n_prev]) - s * (_y[n_next] - _y[n_prev])

    return next_y

def lax_wen():
    """
    Advance 1 step using Lax-Wendroff
    :return:

    """

    pass


def main():

    #Dummy
    x,y = initial_wave(100*dx)

    plt.plot(x,y)

    for i in range(1000):
        y = perfect_shift(y)

    plt.plot(x,y)
    plt.show()
    #end Dummy

    #FTCS
    x, y = initial_wave(100 * dx)
    for i in range(1000):

if __name__ == '__main__':
    main()


