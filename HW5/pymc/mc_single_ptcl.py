#!/usr/bin/env python
from __future__ import print_function
import particleset, simulationcube, pairpotentiallj
import numpy as np
import os
import itertools

# !!!! YOU HAVE TO IMPLEMENT THIS FUNCTION
# make sure to call it at the appropriate place in mc_loop()
def update_position(pset,iat,sigma):
    """ update position of particle "iat" in pset with random step size sigma """
    old_ptcl_pos = pset.pos(iat)

    # this is just a filler line, you have to implement the random move correctly
    new_ptcl_pos = old_ptcl_pos + np.zeros(*old_ptcl_pos.shape)

    return new_ptcl_pos
# end def update_position

# !!!! YOU HAVE TO IMPLEMENT THIS FUNCTION
# make sure to call it at the appropriate place in mc_loop()
def update_position_force(pset,iat,sigma,force_on_iat):
    """ update position of particle "iat" in pset with random step size sigma """
    old_ptcl_pos = pset.pos(iat)

    # this is just a filler line, you have to implement the random move correctly
    new_ptcl_pos = old_ptcl_pos + np.zeros(*old_ptcl_pos.shape)

    return new_ptcl_pos
# end def update_position

# estimators
def compute_energy(sc,pp):
    pset = sc.pset()

    # calculate potential energy
    pot_table = pp.vec_pot(sc.dist_table(copy=False)) # use vectorized version
    tot_potential = pot_table.sum()

    return tot_potential
# end def compute_energy

def compute_particle_energy(iat,sc,pp):
    # calculate potential energy related to a single particle iat
    related_pot_row = pp.vec_pot(sc.dist_table(copy=False)[iat,:])
    related_pot_col = pp.vec_pot(sc.dist_table(copy=False)[:,iat])

    tot_potential = related_pot_row.sum() + related_pot_col.sum()

    return tot_potential
# end def compute_energy

def histogram_distances(sc,num_bin,rmin=0.0,rmax=-1):

    if rmax == -1:
        rmax  = sc.rlimit()
    # end if

    # get all distances
    idx   = np.triu_indices(sc.pset().size(),1)
    dists = sc.dist_table()[idx] 

    rcounts, rbins = np.histogram( dists,bins=num_bin,range=(rmin,rmax) )
    return rcounts, rbins
# end def histogram_distances

def rhok(kvec,pset):
    return np.exp( -1j*np.dot(pset.all_pos(),kvec) ).sum()
# end def rhok

def mc_loop(
    num_atoms   = 64,
    mass        = 48.0,
    temperature = 0.728,
    box_length  = 4.0,
    sigma       = 1.0,
    nsweeps     = 100,
    ndump       = 10,
    naccum      = 20, # sweeps between clearing accumulated variables eg. gofr
    restart     = False,
    trajfile    = 'traj.xyz',
    scalarfile  = 'scalars.dat',
    seed        = 1,
    gofr_file   = 'gofr.dat',
    num_bin     = 20,
    sofk_file   = 'sofk.dat',
    maxk        = 5
    ):
    """ perform Monte Carlo simulation, mindlessly append statistics to scalarfile, and append snapshots of the system to trajfile, file checks should be done outside of this function """

    beta = 1./temperature # inverse temperature
    np.random.seed(seed)  # seed random number generator

    # initialize particle set, no need for velocities
    # ---------------------------
    pset = particleset.ParticleSet(num_atoms,mass)
    sc = simulationcube.SimulationCube(box_length,pset)
    if restart: # existence of trajfile should be checked before md_loop() is called
        assert os.path.exists(trajfile), "restart file %s not found" % trajfile
        pset.load_frame(trajfile)
    else:
        pset.init_pos_cubic(sc.cube_length())
    # end if

    # initialize distances, no need for accelerations
    # ---------------------------
    sc.update_displacement_table()
    pp = pairpotentiallj.PairPotentialLJ(sc)
   
    # setup outputs
    # ---------------------------
    print( "{0:10s}  {1:15s}  {2:15s}".format(
           "Step","Potential","Acceptance") )
    print_fmt = "{isweep:4d}  {potential:15f}  {acceptance:15f}"

    # shortcut for easy printing of numpy arrays
    def arr2str(nparr,fmt="%1.3f"):
        return " ".join([fmt%val for val in nparr])
    # end def

    # get g(r) histogram centers
    gofr    = np.zeros(num_bin)
    rcounts, rbins = histogram_distances(sc,num_bin,rmin=pp.sig()/2.)
    rcenters = (rbins[:-1] + rbins[1:])/2.
    gofr_norm = 4.*np.pi*rcenters**2.*(rcenters[1]-rcenters[0])* pset.size()*(pset.size()-1)/2. /sc.volume()

    # get S(k) k-vectors
    kvecs = (2*np.pi/sc.cube_length())*np.array( [np.array(cube_pos) for cube_pos
        in itertools.product(range(maxk),repeat=pset.ndim())] )
    kmags = np.linalg.norm(kvecs,axis=1)
    ukmags= np.unique(kmags)
    sofk  = np.zeros(len(ukmags))

    if not restart:
        # destroy!
        with open(scalarfile,'w') as fhandle:
            fhandle.write("#     {0:15s}  {1:15s}\n".format(
               "Potential", "Acceptance"))
        # end with open
        with open('gofr.dat','w') as gofr_fhandle:
            gofr_fhandle.write(arr2str(rcenters))
        # end with
        with open('sofk.dat','w') as sofk_fhandle:
            sofk_fhandle.write(arr2str(ukmags))
        # end with
    # end if

    # MC loop
    # ---------------------------
    potential = compute_energy(sc,pp) # only calculate energy of entire system once, then update it as each particle moves for speed
    num_accepted = 0
    for isweep in range(nsweeps):

        # move each particle
        for iat in range(pset.size()):

            # calculate current potential related to this particle
            ptcl_pot = compute_particle_energy(iat,sc,pp)

            # propose a move
            # !!!! YOU HAVE TO CODE THIS

            # calculate new potential related to this particle
            new_ptcl_pot = compute_particle_energy(iat,sc,pp)

            # calculate acceptance rate
            acc_rate = 0.0 # !!!! YOU HAVE TO CODE THIS

            # accept/reject
            if np.random.rand() < acc_rate: # accepted
                pass# !!!! YOU HAVE TO CODE THIS
            else:
                pass# !!!! YOU HAVE TO CODE THIS
            # end if 

        # end for iat

        # accumulate spectral properties

        # gofr
        rcounts, rbins = histogram_distances(sc,num_bin)
        gofr += rcounts

        # sofk
        sk_arr = np.array([(rhok(kvec,pset)*rhok(-kvec,pset)).real for kvec in kvecs])
        for iukmag in range(len(ukmags)):
            # average together whenever kmag is the same 
            kmag = ukmags[iukmag]
            idx2avg = np.where(kmags==kmag)
            sofk[iukmag] += np.mean(sk_arr[idx2avg])
        # end for iukmag
        
        # save snapshot and report
        if (isweep%ndump==0):
            pset.append_frame()
            print( print_fmt.format(**{
                'isweep':isweep,'potential':potential,'acceptance':acc_rate
            }))
        # end if

        # output scalar data
        with open(scalarfile,'a') as fhandle:
            fhandle.write("{0:15f}  {1:15f}\n".format(potential,acc_rate))
        # end with

        # dump spectral properties
        if (isweep % naccum==0 and isweep!=0):

            gofr /= (naccum*gofr_norm) # average and normalize
            with open('gofr.dat','a') as gofr_fhandle:
                gofr_fhandle.write("\n"+" ".join(["%1.3f"%val for val in gofr]))
            # end with
            gofr[:] = 0.0

            sofk /= (naccum*pset.size()) # average and normalize
            with open('sofk.dat','a') as sofk_fhandle:
                sofk_fhandle.write("\n"+" ".join(["%1.3f"%val for val in sofk]))
            # end with
            sofk[:] = 0.0

        # end if

    # end isweep

    print("Acceptance Rate:", float(num_accepted)/nsweeps/pset.size() )

# end def mc_loop
