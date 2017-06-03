#!/usr/bin/env python
from __future__ import print_function
import numpy as np

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
    # calculate new_pos
    
    # map the position back into the simulation box
    for idim in range(ndim):
        if new_pos[idim] >= 0:
            new_pos[idim] -= int(new_pos[idim]/box_length + 0.5) * box_length
        else:
            new_pos[idim] -= int(new_pos[idim]/box_length - 0.5) * box_length
    # end for
    
    return new_pos
# def pos_in_box

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th particle. Unlike the distance function, here you will return a vector instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    # calculate displacement of the iat th particle relative to the jat th particle
    # i.e. r_i - r_j
    # be careful about minimum image convention! i.e. periodic boundary
    ndim = len(posi) # the dimension
    
    # calculate displacement with minimum image convention
    for idim in range(ndim):
        disp[idim] = disp[idim] - posj[idim]
        disp[idim] -= int(disp[idim]/box_length)*box_length
        assert -box_length<disp[idim]<box_length   # the displacement in one dimension shouldn't exceed box_length
        if disp[idim] > 0.5*box_length:
            disp[idim] = disp[idim] - box_length
        elif disp[idim] < -0.5*box_length:
            disp[idim] = disp[idim] + box_length
    # end for
    
    return disp
# end def distance

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum image convention. """

    dist = 0.0
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    dist = np.linalg.norm(displacement(iat, jat, pset, box_length))
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
    ndim = len(pos_t) # dimension
    
    # Verlet
    for idim in range(ndim):
        pos_t_plus_dt[idim] += vel_t[idim]*dt + 0.5*accel_t[idim]*dt*dt
    # end for
    
    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt, 
    based on its velocity at time t, and its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    ndim = len(vel_t)
    
    # Verlet
    for idim in range(ndim):
        vel_t_plus_dt[idim] += 0.5*dt*(accel_t[idim]+accel_t_plus_dt[idim])
    # end for
    
    return vel_t_plus_dt
# end def verlet_next_vel

# We want Lennard-Jones forces. YOU must write this.
# ------------------------------------------------------------------------
def internal_force(iat,pset,box_length):
    """
    We want to return the force on atom 'iat' when we are given a list of 
    all atom positions. Note, pos is the position vector of the 
    1st atom and pos[0][0] is the x coordinate of the 1st atom. It may
    be convenient to use the 'displacement' function above. For example,
    disp = displacement( 0, 1, pset, box_length ) would give the position
    of the 1st atom relative to the 2nd, and disp[0] would then be the x coordinate
    of this displacement. Use the Lennard-Jones pair interaction. Be sure to avoid 
    computing the force of an atom on itself.
    """
    
    force  = np.zeros(pset.ndim())
    natoms = pset.size()   # number of atoms
    ndim   = pset.ndim()   # dimension
    
    # calculate force
    for jat in range(natoms):
        # avoid calculating the mistaken self-exerting force
        if jat == iat:
            continue
        disp   = displacement(iat,jat,pset,box_length) # the displacement of atom i relative to atom j
        dist   = distance(iat,jat,pset,box_length)     # the distance between atom i and j
        ri     = 1.0/dist                 # r_inverse, i.e. 1/r
        r2i    = ri*ri                    # 1/r^2
        r6i    = r2i*r2i*r2i              # 1/r^6
        rforce = 24*r6i*ri * (2*r6i-1)    # radial force on atom i
        
        # add rforce into force
        for dim in range(ndim):
            force[dim] += rforce*disp[dim]*ri
        # end for
    # end for
    return force
# end def internal_force

# calculate the Lennard-Johns potential between iat th atom and jat th atom
def LJ_Potential(iat,jat,pset,box_length):
    dist = distance(iat,jat,pset,box_length) # the distance between the two atoms
    ri   = 1.0/dist                          # r_inverse, i.e. 1/r
    r6i  = ri**6                             # 1/r^6
    
    potential = 4.0*r6i*(r6i-1)
    
    return potential
#end def L-J_Potential

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    vel = pset.all_vel() # all particle velocies
    mass = pset.mass()   # particle mass
    ndim = pset.ndim()   # dimension

    tot_kinetic   = 0.0
    tot_potential = 0.0 

    # calculate total kinetic energy
    for atom in range(natom):
        for idim in range(ndim):
            tot_kinetic += 0.5*mass*vel[atom][idim]*vel[atom][idim]
    # end calculating total kinetic energy
    
    # calculate total potential energy
    for iat in range(natom):
        for jat in range(iat):
            tot_potential += LJ_Potential(iat,jat,pset,box_length)
    # end calculating total potential energy
    
    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

if __name__ == '__main__':

    num_atoms   = 64
    mass        = 48.0
    temperature = 0.728
    box_length  = 4.2323167
    nsteps      = 1000
    dt          = 0.01

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature)

    # molecular dynamics simulation loop
    for istep in range(nsteps):
        
        # update accelerations/forces
        for iat in range(num_atoms):
            iaccel = internal_force(iat,pset,box_length)
            for idim in range(pset.ndim()):
                iaccel[idim] /= mass
            pset.change_accel(iat,iaccel)
        # end for iat

        # calculate properties of the particles
        print(istep, compute_energy(pset,box_length))
        
        # save the configuration of particle 0 for the first 10 steps
        if (istep < 10) :
            outFile=open("config_parti0.dat","a")
            outFile.write(str(istep))
            for index in range(pset.ndim()):
                outFile.write(" "+str(pset.pos(0)[index]))
            outFile.write("\n")
            outFile.close()
        # end if

        # update positions
        for iat in range(num_atoms):
            my_next_pos = verlet_next_pos( pset.pos(iat), pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            pset.change_pos(iat,new_pos)
        # end for iat
        
        # Q/ When should forces be updated?
        new_accel = pset.all_accel()

        # update velocities
        for iat in range(num_atoms):
            my_next_vel = verlet_next_vel( pset.vel(iat), pset.accel(iat), new_accel[iat], dt )
            pset.change_vel( iat, my_next_vel )
        # end for iat

    # end for istep

# end __main__
