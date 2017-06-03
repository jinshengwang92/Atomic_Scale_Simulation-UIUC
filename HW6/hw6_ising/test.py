#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import cubic_ising_lattice
import matplotlib.pyplot as plt

spins_per_side = 20
isingJ         = 1.0
lat = cubic_ising_lattice.CubicIsingLattice(spins_per_side,spin_mag=isingJ)

#lat.flip_spin(100)
#lat.flip_spin(102)
# visualize the lattice
fig = plt.figure()
ax  = fig.add_subplot(111)
lat.visualize(ax)
plt.show()
