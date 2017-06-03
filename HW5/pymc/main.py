#!/usr/bin/env python
from __future__ import print_function
from mc_single_ptcl import mc_loop
import numpy as np
import os

if __name__=="__main__":

    # pass temperature through command line
    import argparse
    parser = argparse.ArgumentParser(description='molecular dynamics of Lennard-Jones fluid with 64 particles with mass 48.0 in a cubic box of length 4.0 using a timestep of 0.032')
    parser.add_argument('-t','--temperature',type=float,default=2.0,help='temperature in reduced units, reasonable range 0.5 - 4.0')
    parser.add_argument('-sig','--sigma',type=float,default=0.03,help='random walk step size')
    parser.add_argument('-n','--nsweeps',type=int,default=200,help='number of simulation steps')
    parser.add_argument('-s','--seed',type=int,default=1,help='random number generator seed')
    parser.add_argument('-r','--restart',action='store_true',help='restart simulation from trajectory file, default to traj.xyz')
    args = parser.parse_args()

    # !!!! change seed when collecting statistics (pass a different -s)
    seed        = args.seed
    temperature = args.temperature      # temperature in reduced units
    nsweeps     = args.nsweeps          # total number of MD steps
    sigma       = args.sigma            # random walk step size
    restart     = args.restart          # restart particle positions from trajectory file

    # Use with caution! May destroy your hard work!
    empty_files = True       # remove all outputs if starting from scratch

    num_atoms   = 64         # number of particles
    mass        = 48.0       # particle mass in reduced units, assume same mass
    box_length  = 4.0        # box length: determines number density with num_atoms

    ndump       = 10         # interval to dump particle positions and report
    naccum      = 5          # steps between clearing accumulated variables eg. gofr
    trajfile    = 'traj.xyz' # trajectory file
    scalarfile  = 'scalars.dat' # scalar file
    gofr_file   = 'gofr.dat' # file to store pair correlation function
    num_bin     = 50         # number of bins for g(r)
    sofk_file   = 'sofk.dat' # file to store structure factor
    maxk        = 5          # max range of k-vector in 1 dimension for S(k)

    # check input & output locations
    # ==============================
    if restart: # restart from trajfile
        # check trajectory file
        if not os.path.exists(trajfile):
            print( "WARNING: no trajectory file found. Starting from scratch" )
            restart = False
        # end if
    # end if restart

    if not restart: # from scratch
        if not empty_files: # stop if any output is about to be overwritten
            for output in [trajfile,scalarfile,gofr_file,sofk_file,velvel_file]:
                if os.path.exists(output):
                    raise RuntimeError("%s exists!" % output)
                # end if
            # end for
        # end if

        # destroy!
        if os.path.exists(trajfile):
            open(trajfile,'w').close()
        # end if
    # end if

    # start MC
    # ==============================
    mc_loop(
        num_atoms   = num_atoms,
        mass        = mass,
        temperature = temperature,
        box_length  = box_length,
        sigma       = sigma,
        nsweeps     = nsweeps,
        ndump       = ndump,
        restart     = restart,
        trajfile    = trajfile,
        scalarfile  = scalarfile,
        gofr_file   = gofr_file,
        sofk_file   = sofk_file,
        seed        = seed,
        naccum      = naccum,
        num_bin     = num_bin,
        maxk        = maxk
    )

# end __main__
