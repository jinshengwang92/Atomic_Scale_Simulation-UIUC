#-*- encoding:utf-8 -*-
from __future__ import print_function
import numpy as np
import random
from numpy import shape
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 2: Introduction to Molecular Dynamics
# ------------------------------------------------------------------------

from particleset import ParticleSet
"""
The ParticleSet class is designed to hold the position, velocity and accelerations of a set of particles. Initialization methods init_pos_cubic(cube_length) and init_vel(temperature) are provided for your convenience.

pset = ParticleSet(natom) will initialize a particle set pset
pset.size() will return the number of particles in pset

| --------------------------------------------------------- | ------------------- |
|                   for access of                           |    use method       |
| --------------------------------------------------------- | ------------------- |
| all particle positions in an array of shape (natom,ndim)  |    pset.all_pos()   |
| all particle velocities                                   |    pset.all_vel()   |
| all particle accelerations                                |    pset.all_accel() |
| particle i position in an array of shape (ndim)           |    pset.pos(i)      |
| particle i velocity                                       |    pset.vel(i)      |
| particle i acceleration                                   |    pset.accel(i)    |
| --------------------------------------------------------- | ------------------- |

| ----------------------------- | ------------------------------------ |
|           to change           |             use method               |
| ----------------------------- | ------------------------------------ |
| all particle positions        |  pset.change_all_pos(new_pos_array)  |
| particle i position           |  pset.change_pos(i,new_pos)          |
| ditto for vel and accel       |  pset.change_*(i,new_*)              |
| ----------------------------- | ------------------------------------ |
"""

# Routines to ensure periodic boundary conditions that YOU must write.
# ------------------------------------------------------------------------
def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary conditions. The simulation box is a cube of size box_length^ndim centered around the origin vec{0}. """

    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    for i in range(ndim):
        new_pos[i] -= box_length*int(round(new_pos[i]/box_length))
    # calculate new_pos
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    ndim = pset.ndim()      #ndim = pset.shape[0]
    for i in range(ndim):
        disp[i]  = posi[i] - posj[i]
        disp[i] -= box_length*int(round(disp[i]/box_length))
    # calculate displacement of the iat th particle relative to the jat th particle
    # i.e. r_i - r_j
    # be careful about minimum image convention! i.e. periodic boundary
    return disp
# end def distance

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum image convention. """
    dist = 0.0
    vector = displacement(iat, jat, pset, box_length)
    dist = np.linalg.norm(vector)
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    return dist

# end def distance

# The Verlet time-stepping algorithm that YOU must write, dt is time step
# ------------------------------------------------------------------------
def verlet_next_pos(pos_t,vel_t,accel_t,dt):
    """
    We want to return position of the particle at the next moment t_plus_dt
    based on its position, velocity and acceleration at time t.
    """
    pos_t_plus_dt = pos_t.copy()
    pos_t_plus_dt += dt*(vel_t+dt*accel_t/2)
    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt,
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    vel_t_plus_dt += dt*accel_t/2 + dt*accel_t_plus_dt/2
    return vel_t_plus_dt
# end def verlet_next_vel

# We want Lennard-Jones forces. YOU must write this.
# ------------------------------------------------------------------------
def internal_force(iat,pset,box_length):
    """
    We want to return the force on atom 'iat' when we are given a list of
    all atom positions. Note, pos is the position vector of all the
    atom and pos[0][0] is the x coordinate of the 1st atom. It may
    be convenient to use the 'displacement' function above. For example,
    disp = displacement( 0, 1, pset, box_length ) would give the position
    of the 1st atom relative to the 2nd, and disp[0] would then be the x coordinate
    of this displacement. Use the Lennard-Jones pair interaction. Be sure to avoid
    computing the force of an atom on itself.
    """

    pos = pset.all_pos()  # positions of all particles
    mypos = pset.pos(iat) # position of the iat th particle
    force = np.zeros(pset.ndim())  # allocate the space for force
    for i in range(num_atoms):
        if i != iat:
            dis = distance(iat, i, pset, box_length)
            x_inverse2 = 1/(dis*dis)
            x_inverse6 = x_inverse2*x_inverse2*x_inverse2
            disp = displacement(iat, i, pset, box_length)
            force += 24*x_inverse6*x_inverse2*(2*x_inverse6-1)*disp
        else:
            continue
    # calculate force
    return force
# end def internal_force

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    pos = pset.all_pos() # all particle positions
    vel = pset.all_vel() # all particle velocies

    tot_kinetic = 0.5*mass*(vel**2).sum()
    tot_potential = 0.0
    for i_par in range(natom):
        for j_par in range(i_par+1,natom):
            r_inverse = 1/distance(i_par,j_par, pset, box_length)
            if 1/r_inverse <= box_length*np.sqrt(3)/2:   #np.sqrt(3)/2*box_length
                r_inverse_6 = r_inverse*r_inverse*r_inverse*r_inverse*r_inverse*r_inverse
            #cutoff of potential is not judged here!
                tot_potential += 4*(r_inverse_6*r_inverse_6 - r_inverse_6)
            else:
                continue
    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

if __name__ == '__main__':

    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    box_length  = 4.2323167
    nsteps      = 1000  # total step for simulation
    energy_all  = np.zeros((nsteps,3))  # used to store energy
    pos_par_0   = np.zeros((nsteps,3))  # used to store the postions of particle 0
    dt          = 0.01  #time step
    seed = 20   # change seed to initial velocity

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset._alloc_pos_vel_accel()
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature,seed)
    # molecular dynamics simulation loop
    for istep in range(nsteps):

        # calculate properties of the particles
        Ene_all = compute_energy(pset,box_length)
        print(istep, Ene_all[0],Ene_all[1],Ene_all[2])
        energy_all[istep][0] = Ene_all[0]
        energy_all[istep][1] = Ene_all[1]
        energy_all[istep][2] = Ene_all[2]

        # update positions
        for iat in range(num_atoms):
            old_acc = internal_force(iat,pset,box_length)/mass
            pset.change_accel(iat,old_acc)
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            if iat == 0:             # find and store the position for particle 0
                pos_par_0[istep,:] = new_pos
            pset.change_pos(iat,new_pos)
            my_new_acc = internal_force(iat,pset,box_length)/mass
            pset.change_accel(iat,my_new_acc)
            my_next_vel = verlet_next_vel( pset.vel(iat), old_acc, pset.accel(iat), dt )
            pset.change_vel( iat, my_next_vel )


    # to get the postions of the particel 0
    #print(pos_par_0)

    # print the mean and stddev of TE on screen for later use
    print('*'*40)
    print('the mean of TE = %.8f'%np.mean(energy_all[:,2]))
    print('the stddev of TE = %.8f'%np.std(energy_all[:,2]))
    print('*'*40)
    #end of print of mean and stddev


    #plot for figure TE,PE,KE
    plt.figure()
    x_axis = range(0,nsteps)
    y_KE = energy_all[:,0]
    y_PE = energy_all[:,1]
    y_TE = energy_all[:,2]
    plt.plot(x_axis,y_KE,label='Kinetic Energy')
    plt.plot(x_axis,y_PE,label='Potential Energy')
    plt.plot(x_axis,y_TE,label='Total Energy')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Energy')
    plt.title(r'Plot of energy when dt=%.2f, %d steps'%(dt,nsteps))
    plt.show()
    #end for plot of figure



# end __main__
