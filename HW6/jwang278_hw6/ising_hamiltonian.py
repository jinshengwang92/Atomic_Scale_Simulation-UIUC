from __future__ import print_function
import numpy as np
from copy import deepcopy


class IsingHamiltonian:

    def __init__(self,J,lat):
        self._isingJ   = J # OK to use one-letter variable once in constructor
        self._lat_ref = lat # keep a reference to an IsingLattice
    # end def

    def isingJ(self):
        return self._isingJ
    # end def isingJ

    def lattice(self,copy=True):
        if copy:
            return deepcopy(self._lat_ref)
        else:
            return self._lat_ref
        # end if
    # end def lattice

    def compute_energy(self):
        """ calculate the total energy of the tracked ising lattice
        note: the state of the spins (up/down) is stored in lattice.spins(), and the magnitude of the spins are stored separately in lattice.spin_mag() """
        tot_energy = 0.0
        latt       = self.lattice(copy=False) # do NOT edit lattice
        N_total    = latt.size()
        mag        = latt.spin_mag()
        Jfactor    = self.isingJ()
        for ispin in range(N_total):
            nb   = latt.nb_list(copy=False)[ispin]
            here = latt.spin(ispin)
            tot_energy += 0.5*here*(latt.spin(nb[0])+latt.spin(nb[1])+latt.spin(nb[2])+latt.spin(nb[3]))
            #each spin is calculated 4 times, thus divided by 4 here
        # !!!! IMPLEMENT THIS FUNCTION. done !!
        tot_energy *= -mag*mag*Jfactor
        return tot_energy
    # end def compute_energy

    def compute_spin_energy(self,ispin):
        """ calculate the energy change associated with one spin flip
        note: the state of the spins (up/down) is stored in lattice.spins(), and the magnitude of the spins are stored separately in lattice.spin_mag() """
        # store some references for easy access
        latt       = self.lattice(copy=False) # do NOT edit lattice
        nb         = latt.nb_list(copy=False)[ispin] # do NOT edit neighbours!
        here       = latt.spin(ispin)
        mag        = latt.spin_mag()
        Jfactor    = self.isingJ()
        energy     = here*(latt.spin(nb[0])+latt.spin(nb[1])+latt.spin(nb[2])+latt.spin(nb[3]))
        del_ene    = 2.0*mag*mag*Jfactor*energy   # energy change due to ispin flip
        # !!!! IMPLEMENT THIS FUNCTION. done!!
        # calculate energy change
        return del_ene
    # end def compute_spin_energy

# end class IsingHamiltonian

if __name__ == '__main__':
    from cubic_ising_lattice import CubicIsingLattice

    # 16 spins -> 32 n.n. bonds -> each bond holds -0.25 energy -> -8 total
    lat = CubicIsingLattice(4,spin_mag=0.5)
    ham = IsingHamiltonian(1.0,lat)
    #print(ham.compute_energy())
    assert np.isclose(-8,ham.compute_energy())

    # turn 4 bonds to 4 anti-bonds -> cost -0.25*4*2=-2.0 energy
    print(ham.compute_spin_energy(0))
    lat.flip_spin(0)
    assert np.isclose(-6,ham.compute_energy())
# end if
