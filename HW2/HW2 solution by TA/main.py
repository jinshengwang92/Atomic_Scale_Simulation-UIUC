#!/usr/bin/env python
from __future__ import print_function
import particleset, simulationcube, pairpotentiallj
import numpy as np
import os

# vectorized velocity verlet integrators
def verlet_next_pos(pset,dt):
    return pset.all_pos() + pset.all_vel()*dt + 0.5*pset.all_accel()*dt**2.
# end def
def verlet_next_vel(pset,all_old_accel,dt):
    return pset.all_vel() + 0.5*(pset.all_accel()+all_old_accel)*dt
# end def

# estimators
def compute_energy(sc,pp):
    pset = sc.pset()

    # calculate kinetic energy
    tot_kinetic = 0.5*pset.mass()*(pset.all_vel()**2.).sum()

    # calculate potential energy
    pot_table =  sc.dist_table(copy=True)
    for iat in range(pset.size()):
        for jat in range(iat+1,pset.size()):
            pot_table[iat,jat] = pp.pot( pot_table[iat,jat] )
        # end for jat
    # end for iat
    #np.apply_over_axes(pp.pot,sc.dist_table(copy=False),(0,1)) 
    tot_potential = pot_table.sum()

    tot_energy = tot_kinetic + tot_potential

    return (tot_kinetic, tot_potential, tot_energy)
# end def compute_energy

def md_loop(
    num_atoms   = 64,
    mass        = 48.0,
    temperature = 0.728,
    box_length  = 4.2323167,
    dt          = 0.01,
    nsteps      = 1000,
    ndump       = 1,
    restart     = False,
    trajfile    = 'traj.xyz',
    seed        = 1
    ):

    # initialize particle set
    # ---------------------------
    pset = particleset.ParticleSet(num_atoms,mass)
    sc = simulationcube.SimulationCube(box_length,pset)

    if not restart: # from scratch
        pset.init_pos_cubic(sc.cube_length())
        pset.init_vel(temperature,seed=seed)
        # check trajectory file
        if os.path.exists(trajfile): # empty trajfile
            fhandle = open(trajfile,'w')
            fhandle.write('')
            fhandle.close()
        # end if
    else: # restart
        # check trajectory file
        if not os.path.exists(trajfile):
            print( "WARNING: no trajectory file found. Starting from scratch" )
            pset.init_pos_cubic(sc.cube_length())
            pset.init_vel(temperature)
        else:
            pset.load_frame(trajfile)
            pset.init_vel(temperature) # should really load past velocities...
            print( "WARNING: restarting with random velocities" )
        # end if
    # end if

    # initialize distances and accelerations
    # ---------------------------
    sc.update_displacement_table()
    pp = pairpotentiallj.PairPotentialLJ(sc,rmin=1e-2)
    all_old_accel = pp.new_accel()

    # MD loop
    # ---------------------------
    #pos_trace = np.zeros([nsteps,pset.all_pos().shape[0],pset.all_pos().shape[1]])
    print( "{0:10s}  {1:15s}  {2:15s}  {3:15s}".format("Step","Kinetic","Potential","Total") )
    print_fmt = "{istep:4d}  {kinetic:15f}  {potential:15f}  {total:15f}"
    for istep in range(nsteps):
	
	#pos_trace[istep] = pset.all_pos(copy=True)
        if (istep%ndump==0):
            kinetic, potential, total = compute_energy(sc,pp)
            print( print_fmt.format(**{
                'istep':istep,'kinetic':kinetic
                ,'potential':potential,'total':total
            }))
            pset.append_frame()
        # end if

        pset.change_all_pos( sc.pos_in_box( verlet_next_pos(pset,dt) ) )
        sc.update_displacement_table()

        pset.change_all_accel( pp.new_accel() )
        pset.change_all_vel( verlet_next_vel(pset,all_old_accel,dt ) )
        all_old_accel = pset.all_accel()

    # end for istep
# end def md_loop

if "__main__":

    num_atoms   = 64         # number of particles
    mass        = 48.0       # particle mass in reduced units, assume same mass
    temperature = 0.728      # temperature in reduced units
    box_length  = 4.2323167  # box length: this determines number density
    dt          = 0.01       # MD time step
    nsteps      = 1000       # total number of MD steps
    ndump       = 10         # interval to dump particle positions
    restart     = False      # restart particle positions from trajectory file
    trajfile    = 'traj.xyz' # trajectory file
    seed        = 1          # pseudo random number generator seed
    # !!!! change seed when collecting statistics

    md_loop(num_atoms,mass,temperature,box_length,dt,nsteps
        ,ndump,restart,trajfile,seed)

# end __main__
