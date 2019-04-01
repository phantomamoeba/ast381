"""
Collection of utilities such as random selection of direction, conversion between coord systems, validations plotting
"""
import numpy as np
import matplotlib.pyplot as plt

def prand_seed(seed):
    """
    Update the pRNG seed
    :param seed:
    :return:
    """
    np.random.seed(seed)


def cm2pc(cm):
    """
    cm to parsecs
    :param cm:
    :return:
    """
    return cm/3.086e18

def pc2cm(pc):
    """
    parsecs to cm
    """
    return pc * 3.086e18

def get_uniform_prob(seed=None):
    """
    Return a single, uniform random # 0-1
    :param seed: if provided, will (re) seed the pRNG
    :return: number [0-1]
    """

    if seed is not None:
        np.random.seed(seed)

    return np.random.random()

def get_direction(seed=None):
    """
    return a single coord tuple (theta, phi) where theta 0-pi and phi 0-2pi
    :param seed: if provided, will (re) seed the pRNG
    :return:
    """

    if seed is not None:
        np.random.seed(seed)

    #generate random uniform -1 to 1 as mu = cos(theta) ... so theta == arccos(mu)
    theta = np.arccos(2. * np.random.random() -1.)
    #phi runs 0 - 2pi
    phi = 2. * np.pi * np.random.random()

    return theta,phi


def get_interaction_dist(seed=None):
    """
    Return a randomly sampled distance to the (next) scattering encounter

    :param seed:
    :return: distance travel in cm
    """
    if seed is not None:
        np.random.seed(seed)

    t = -1. * np.log(np.random.random())
    s_Th = 0.67e-24 #Thomson cross section (cm^2)
    n_e = 60000. #electron number density, same as n_H = p_H / m_H ~ 59880.23952095808

    return t/(s_Th * n_e)


def sphere2cart(r,t,p):
    """
    Convert spherical polar coords to Cartessian
    :param r: radius
    :param t: polar angle (0-pi)
    :param p: azimuthal angle (0-2pi)
    :return: x,y,z
    """

    x = r * np.sin(t) * np.cos(p)
    y = r * np.sin(t) * np.sin(p)
    z = r * np.cos(t)

    return x,y,z



def prob_absorb(w):
    """

    Goes to constant ~ 20% for wavelengths < 0.05micron.
    Approaches 0% on longer wavelegnths (0.06% by 1 micron).

    :param w: wavelength in microns
    :param density:
    :return:
    """

    density_H = 1e-19 #g cm^-3
    density_dust = 1e-3 * density_H
    k_Th = (0.67e-24)/(1.67e-24) #cm^2/g Thomson opacity
    k_0 = 100. #cm^2 g^-1
    w_0 = 0.05 #microns
    k_d = k_0 * (w/w_0)**(-2.) if (w > w_0) else k_0

    return density_dust*k_d / (density_dust * k_d + density_H * k_Th)




def test_interaction_dist(count=10000,bins=50,fn=None):
    """
    Plot histogram of interaction distances. Should be an exponential decay (like optical depth)
    :param count: number of distances to get
    :param bins: number of bins
    :param fn: savefig to file, if provided
    :return:
    """
    d = []
    for _ in range(count):
        d.append(get_interaction_dist())

    d = np.array(d)
    d /= 3.086e18



    plt.title("Interaction Distances (pc)")
    plt.ylabel("count")
    plt.xlabel("distance [pc]")
    plt.hist(d, bins=bins)
    plt.tight_layout()

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)


def test_direction(count=10000,bins=50,fn=None):
    """
    Plot histogram of directions to show uniform distribution over theta (as cos(mu)) and phi
    :param count: number of directions to get
    :param bins: number of bins
    :param fn: savefig to file, if provided
    :return:
    """
    theta = []
    phi = []

    for _ in range(count):
        t,p = get_direction()
        theta.append(t)
        phi.append(p)

    theta = np.array(theta)
    phi = np.array(phi)

    plt.figure(figsize=(10, 3))
    plt.subplots_adjust(wspace=0.4, hspace=0.4)


    plt.subplot(131)
    plt.suptitle("")
    plt.title(r"$\theta$")
    plt.ylabel("count")
    plt.xlabel("radians")
    plt.hist(theta,bins=bins)


    plt.subplot(132)
    plt.suptitle("")
    plt.title(r"$\mu$ = cos($\theta$)")
    plt.ylabel("count")
    plt.xlabel("radians")
    plt.hist(np.cos(theta),bins=bins)

    plt.subplot(133)
    plt.suptitle("")
    plt.title(r"$\phi$")
    plt.ylabel("count")
    plt.xlabel("radians")
    plt.hist(phi,bins=bins)

    plt.tight_layout()

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)


def test_prob_absorb(fn=None):
    """
    Show the prob of absorption curve.

    :param fn: savefig to file, if provided
    :return:
    """

    p = []
    w = np.logspace(-3, 2, 10000)
    for x in w:
        p.append(prob_absorb(x))

    plt.title("Probability (0-1) of Absorption by Wavelength")
    plt.plot(w,p)
    plt.xscale("log")
    plt.ylabel("Prob of Absorption")
    plt.xlabel(r"$\lambda$ [microns")
    plt.tight_layout()

    if fn is None:
        plt.show()
    else:
        plt.savefig(fn)