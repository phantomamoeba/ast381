#Dustin Davis
#AST 381 Computational Astrophysics
#Homework #6
#April 18, 2019

__author__ = "Dustin Davis"

"""
intro comments
"""


import numpy as np
import tables
import os.path as op
import matplotlib.pyplot as plt

mid_x = 40.0

BASEDIR = "/home/dustin/code/python/ast381_compastro/hw6/"


class Particle:
    def __index__(self):
        self.x = -1.
        self.vx = 0.
        self.density = 0.
        self.pressure = 0.

    def clean_noise(self):
        if self.vx < 1e-17:
            self.vx = 0.0

        if self.density < 1e-17:
            self.density = 0.0

        if self.pressure < 1e-17:
            self.pressure = 0.0

#testing
if False:
    h5 = tables.open_file(BASEDIR+"out/snapshot_000.hdf5")
    coords = h5.root.PartType0.Coordinates
    x = [c[0] for c in coords]
    vels = h5.root.PartType0.Velocities
    vx = [v[0] for v in vels]
    densities = h5.root.PartType0.Density
    internal_e = h5.root.PartType0.InternalEnergy



def read_snapshot(fn):

    h5 = tables.open_file(BASEDIR+fn)

    particles = []

    # coords = h5.root.PartType0.Coordinates
    # x = [c[0] for c in coords]
    # vels = h5.root.PartType0.Velocities
    # vx = [v[0] for v in vels]
    # densities = h5.root.PartType0.Density
    # internal_e = h5.root.PartType0.InternalEnergy

    for i in range(len(h5.root.PartType0.Coordinates)):
        p = Particle()
        p.x = h5.root.PartType0.Coordinates[i][0] - mid_x
        p.vx = h5.root.PartType0.Velocities[i][0]
        p.density = h5.root.PartType0.Density[i]
        p.pressure = h5.root.PartType0.InternalEnergy[i]
        p.clean_noise()
        particles.append(p)

    h5.close()

    particles.sort(key=lambda a:a.x )


    return particles

def make_plot(particles,fn=None):

    plt.close('all')

    plt.figure()
    plt.subplots_adjust(hspace=0.4)

    x = [p.x for p in particles]

    plt.subplot(411)
    plt.xlim((-10, 10))
    plt.plot(x, [p.density for p in particles])
    plt.ylabel(r"$\rho$")

    plt.subplot(412)
    plt.xlim((-10, 10))
    plt.plot(x, [p.pressure for p in particles])
    plt.ylabel("P")

    plt.subplot(413)
    plt.xlim((-10, 10))
    plt.ylim((0,0.2))
    plt.plot(x, [p.vx for p in particles])
    plt.ylabel(r"$v_{x}$")

    plt.subplot(414)
    plt.xlim((-10, 10))
    plt.plot(x, np.array([p.pressure for p in particles])/(np.array([p.density for p in particles])**1.4))
    plt.ylabel(r"P/$\rho^{\gamma}$")

    plt.xlabel("x")

    if fn is not None:
        plt.savefig(BASEDIR+fn)
    else:
        plt.show()



def main():

    make_plot(read_snapshot("out/snapshot_000.hdf5"), fn="t00.png")
    make_plot(read_snapshot("out/snapshot_001.hdf5"), fn="t01.png")

    make_plot(read_snapshot("out/snapshot_002.hdf5"), fn="t02.png")
    make_plot(read_snapshot("out/snapshot_003.hdf5"), fn="t03.png")
    make_plot(read_snapshot("out/snapshot_004.hdf5"), fn="t04.png")

    make_plot(read_snapshot("out/snapshot_005.hdf5"), fn="t05.png")

    make_plot(read_snapshot("out/snapshot_006.hdf5"), fn="t06.png")
    make_plot(read_snapshot("out/snapshot_007.hdf5"), fn="t07.png")
    make_plot(read_snapshot("out/snapshot_008.hdf5"), fn="t08.png")
    make_plot(read_snapshot("out/snapshot_009.hdf5"), fn="t09.png")


    make_plot(read_snapshot("out/snapshot_010.hdf5"), fn="t10.png")




if __name__ == '__main__':
    main()