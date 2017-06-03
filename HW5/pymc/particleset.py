#!/usr/bin/env python
from __future__ import print_function

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------

import numpy as np
import itertools
from copy import deepcopy

class ParticleSet:

    """ The ParticleSet class is designed to hold the position, velocity and accelerations of a set of particles. The self._pos, self._vel, self._accel variables are NOT meant to be accessed or modified directly. Please use accessor and modifier methods instead. 
    Initialization methods init_pos_cubic(cube_length) and init_vel(temperature) are provided for your convenience. """

    def __init__(self,num,mass,ndim=3,name="atoms"):
        """ create a particle set with num atoms in ndim dimensions. """
        # allocate local variables and arrays
        # particles still need to be initialized before the particle set can be used
        self.name  = name           # name of particle set, only used in __str__ 
        self._mass = mass           # mass of the particles, !!!! assume same for every particle
        self._nat  = num            # number of atoms
        self._ndim = ndim           # spatial dimension
        self._alloc_pos_vel_accel() # allocate memory for position, velocity and acceleration  
        self.initialized = False   # must call a position initialization routine before this particle set object can be used in a simulation, set it to true at your own risk of having the simulation blow up
    # end def

    def __str__(self):
        # allow ParticleSet object to be printed
        description = "ParticleSet " + self.name + " has %d particles \n" % self._nat
        description += str(self._pos) 
        if not self.initialized:
            description += "\n WARNING: not initialized "
        # end if
        return description
    # end def

    def _alloc_pos_vel_accel(self):
        # allocate arrays for position, velocity and acceleration
        self._pos   = np.zeros((self._nat,self._ndim))
        self._vel   = np.zeros((self._nat,self._ndim))
        self._accel = np.zeros((self._nat,self._ndim))
    # end def

    # ---------------------------
    # begin accessor methods
    #  these are trivial accessor methods
    # Note: These methods can be generalized better, but I chose to exapand them for readability.

    def size(self):
        return self._nat
    # end def size

    def name(self):
        return self.name
    # end def name

    def ndim(self):
        return self._ndim
    # end def ndim

    def mass(self):
        return self._mass
    # end def mass

    def all_pos(self,copy=True):
        # return the positions of all the atoms
        if copy:
            return deepcopy( self._pos )
        else:
            # fast but dangerous!
            return self._pos
        # end if 
    # end def 

    def all_vel(self,copy=True):
        # return the velocities of all the atoms
        if copy:
            return deepcopy( self._vel )
        else:
            # fast but dangerous!
            return self._vel
        # end if 
    # end def 

    def all_accel(self,copy=True):
        # return the accelerations of all the atoms
        if copy:
            return deepcopy( self._accel )
        else:
            # fast but dangerous!
            return self._accel
        # end if
    # end def 

    def pos(self,iat):
        # return position of the iat th atom
        return self._pos[iat,:].copy()
    # end def

    def vel(self,iat):
        # return velocity of the iat th atom
        return self._vel[iat,:].copy()
    # end def

    def accel(self,iat):
        # return acceleration of the iat th atom
        return self._accel[iat,:].copy()
    # end def

    # end accessor methods
    # ---------------------------

    # ---------------------------
    # begin modifier methods
    #  these are trivial modifier methods

    # use numpy's built-in functions for fast array assignment
    def change_all_pos(self,new_pos):
        # guard against misuse
        assert new_pos.shape == self._pos.shape, "shape mismatch"
        # renew all entries of self._pos fast
        np.copyto(self._pos,new_pos)
    # end def

    def change_all_vel(self,new_vel):
        assert new_vel.shape == self._vel.shape, "shape mismatch"
        np.copyto(self._vel,new_vel)
    # end def

    def change_all_accel(self,new_accel):
        assert new_accel.shape == self._accel.shape, "shape mismatch"
        np.copyto(self._accel,new_accel)
    # end def

    def change_pos(self,iat,pos1):
        assert pos1.shape==self._pos[iat,:].shape, "shape mismatch"
        self._pos[iat,:] = pos1
    # end def

    def change_vel(self,iat,vel1):
        assert vel1.shape==self._vel[iat,:].shape, "shape mismatch"
        self._vel[iat,:] = vel1
    # end def

    def change_accel(self,iat,accel1):
        assert accel1.shape==self._accel[iat,:].shape, "shape mismatch"
        self._accel[iat,:] = accel1
    # end def

    # end modifier methods
    # ---------------------------

    # ---------------------------
    # begin initializer methods
    #  these functions do NOT require modification

    # Note: Everyone should start their gas in the same initial configuration!

    def init_pos_cubic(self,cube_length):
        # initialize the particles in uniform cubic array

        # determine how many atoms should be on each side of the cube
        nat_per_side = int(round( (self._nat)**(1./3) ))
        if (nat_per_side**3 != self._nat ):
            raise NotImplementedError("init_pos_cubic() is not intented to be used for when the total number of particles (" + str(self._nat) + ") is not a perfect cube.")
        # end if

        # generate integer positions for the atoms inside the cube
        new_pos = np.array([np.array(cube_pos) for cube_pos 
            in itertools.product(range(nat_per_side),repeat=self._ndim) ]) 

        # make sure generated positions have the right shape

        # atom separation
        rsep = float(cube_length)/nat_per_side
        new_pos = rsep * new_pos

        # shift atoms towards the center of the box, assuming origin is at (0,0,0)
        new_pos -= (cube_length-rsep)/2.*np.ones(self._ndim)
        # - cube_length/2. centers the particles around (0,0,0), but there are particles at the edge of the box
        # + rsep/2. gives every particle equal space from the edges

        # renew all entries of self._pos fast
        self.change_all_pos(new_pos)

        self.initialized = True

    # end def init_pos_cubic

    def init_vel(self, temperature, seed=1):

        np.random.seed(seed)

        new_vel = np.random.rand(self._nat,self._ndim) - 0.5       # random velocities
        com_vel = new_vel.sum(axis=0)/self._nat # center of mass velocity
        new_vel -= com_vel # don't let center of mass move, or soon empty box

        # renormalize velocities
        sumv2   = (new_vel**2).sum()
        new_vel /= np.sqrt(sumv2)

        # change velocity magnitude to reflect temperature
        vel_mag = np.sqrt(3.*self._nat*temperature/self.mass())
        new_vel *= vel_mag

        self.change_all_vel(new_vel)

    # end def init_vel

    # end initializer methods
    # ---------------------------

    # ---------------------------
    # begin file I/O methods

    def append_frame(self,filename="traj.xyz"):
        fhandle = open(filename,"a")
        fhandle.write(str(self._nat) + "\n\n" +\
                "\n".join( [ "He  " + " ".join(vec.astype(str)) for vec in self._pos] ) + "\n")
        fhandle.close()
    # end def append_frame

    def load_frame(self,filename="traj.xyz",ref_frame=-1,max_nframe=10**6):
        """ initialize particle positions from the last frame of a previous simulation """ 
        from mmap import mmap
        fhandle = open(filename,'r+')
        mm = mmap(fhandle.fileno(),0)

        # first line should specify the number of atoms
        line  = mm.readline()
        natom = int(line)
        mm.seek(0)
        if natom != self._nat:
            raise InputError("number of atoms in the trajectory file (%d) does not match the number of atoms in this particle set %d"%(natom,self._nat))
        # end if

        # find starting positions of each frame in trajectory file
        frame_starts = []
        for iframe in range(max_nframe):
            # stop if end of file is reached 	
            if mm.tell() >= mm.size():
                break
            # end if

            # locate starting line (should be "%d\n" % natom)
            idx = mm.find(line)
            if idx == -1:
                break
            # end if
            mm.seek(idx)
            myline = mm.readline()
            next_line = mm.readline()
            if len(next_line.split()) != 0:
                continue # some coordinate ended with "64\n"
            # end if

            frame_starts.append(idx)
        # end for iframe

        # go to desired frame
        mm.seek(frame_starts[ref_frame])

        # skip two lines (current and comment lines)
        for i in range(2):
            mm.readline()
        # end for

        # read atomic positions
        for iat in range(natom):
            self._pos[iat,:] = np.array(mm.readline().split()[-self._ndim:],dtype=float)
        # end for

        mm.close()
        fhandle.close()

    # end def load_frame

    # end file I/O methods
    # ---------------------------


# end class ParticleSet

if __name__ == '__main__':

    num_atoms   = 8
    mass        = 48.0
    temperature = 0.728
    box_length  = 1.0

    # initialize particle set
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    assert pset.size() == num_atoms, "wrong number of atoms in particle set " + pset.name()

    # visual inspection
    print( pset )

    # initialize velocities
    pset.init_vel(temperature)
    vel = pset.all_vel()

    # check that velocities are initialized to the correct temperature
    tot_kinetic = 0.5*mass*(vel**2).sum()
    Uinternal   = 1.5*num_atoms*temperature
    assert abs(tot_kinetic-Uinternal) < 1e-10, "total kinetic energy = %f is different from internal energy" 

    # make sure accessor cannot be used as modifier
    original_pos0 = pset.pos(0).copy()
    new_pos0      = pset.pos(0)
    new_pos0[0]   += 10.0
    assert  np.allclose(pset.pos(0),original_pos0)

# end __main__
