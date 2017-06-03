#!/usr/bin/env python
from __future__ import print_function
import cubic_ising_lattice, ising_hamiltonian
import numpy as np
import os
from copy import deepcopy

def check_in(lat,ref_s,cluster_before,last_add,acc_r):
    #function for cluster iteration
    old_cluster = deepcopy(cluster_before)  # old all spins
    old_add     = deepcopy(last_add)     #spins added last time
    new_add     = []   # to store the new spin this time
    for ii in old_add:
        for jj in lat.neighbours(ii):
            if (jj not in old_cluster) and lat.spin(jj) == ref_s:
                if np.random.random()< acc_r:  # add into cluster with probability
                    cluster_before += [jj]   #upadate all spins
                    new_add        += [jj]   #update new spins this time
                #end if
            #end if
        #end if
    #end for
    if new_add == []:  # cluster remains not change, end loop
        return old_cluster
    else:  #if new cluster is different than old ,have to loop again
        return check_in(lat,ref_s,cluster_before,new_add,acc_r) # in fact this would be a nested loop
#end def of check_in

def build_cluster(lat,beta,IJ):
    #returns list of Cluster
    cluster = []
    ispin = np.random.randint(lat.size())  # choose a random position
    print('start ispin = '+str(ispin))
    cluster += [ispin]
    ref_s = lat.spin(ispin)  # the reference spin
    mag   = lat.spin_mag()
    acc_r = 1.0 - np.exp(-beta*IJ*(1.0+mag*mag))
    clust   =  check_in(lat,ref_s,cluster,[ispin],acc_r)  # this is a funciton defined above
    ans     =  list(set(clust))
    return ans   # final cluster
#end def of cluster

def flip_spins(lat,cluster):
    # take the cluster and flips all the spins in the cluster
    #return how much the magnetization has changed.
    mag    = lat.spin_mag()
    ref_s  = lat.spin(cluster[0])
    dmag   = 0.0
    for ii in cluster:  # flip all the spins
        lat.flip_spin(ii)
    #end for
    #dmag   = -2.0*ref_s*mag*len(cluster)  # in fact, I did not use dmag for later use
    return dmag
#end def of flip_spins


def mc_loop(
    spin_mag       = 1.0,
    isingJ         = 1.0,
    temperature    = 4.0,
    spins_per_side = 20,
    nsweep         = 100,
    ndump          = 25,
    seed           = 1,
    traj_file      = 'traj.dat',
    scalar_file    = 'scalar.dat',
    restart        = False,
    ):
    """ perform Monte Carlo simulation of the Ising model """

    beta = 1.0/temperature
    np.random.seed(seed)

    # initialize the Ising model
    # ---------------------------
    lat = cubic_ising_lattice.CubicIsingLattice(spins_per_side,spin_mag=spin_mag)
    print(lat)
    ham = ising_hamiltonian.IsingHamiltonian(isingJ,lat)
    if restart: # check traj_file
        assert os.path.exists(traj_file), "restart file %s not found" % traj_file
        lat.load_frame(traj_file)
    # end if restart

    # setup outputs
    # ---------------------------
    if not restart:
        # destroy!
        with open(scalar_file,'w') as fhandle:
            fhandle.write('# Energy    M^2/spin\n')
        # end with open
    # end if
    print_fmt   = "{isweep:4d}  {energy:10.4f}  {Mperspin2:9.6f}"
    print(" isweep   energy   M2 ")

    # initialize observables
    # ---------------------------
    tot_energy  = ham.compute_energy()
    tot_mag     = lat.magnetization()
    m_per_spin2 = (tot_mag*1.0/lat.size())**2

    # MC loop
    # ---------------------------
    num_accept  = 0.0
    for isweep in range(nsweep):

        # report and dump configuration
        if isweep%ndump==0:
            # report
            print(print_fmt.format(**{'isweep':isweep,'energy':tot_energy
                ,'Mperspin2':m_per_spin2}))
            # checkpoint
            if isweep==0 and restart:
                pass # do not duplicate last configuration
            else:
                lat.append_frame(traj_file)
            # end if
        # end if

        # save scalar data
        with open(scalar_file,'a') as fhandle:
            fhandle.write('{energy:10.4f}  {Mperspin2:9.6f}\n'.format(**{'energy':tot_energy,'Mperspin2':m_per_spin2}))
        # end with open

        ''' #===============   Cluster  MC loop =================='''
        for imove in range(lat.size()):
            cluster = build_cluster(lat,beta,isingJ)  #build a cluster
            print('cluster        = '+str(cluster))
            print('cluster Length = '+str(len(cluster)))
            flip_spins(lat,cluster)         #flip the cluster, the magnetization change is not used
            # the magnetization will be recalculated each sweep, the magnetization change method for each imove is
            # not used since the speed difference is nuance
        tot_energy  = ham.compute_energy()
        tot_mag     = lat.magnetization()
        m_per_spin2 = (tot_mag/lat.size())**2


        # end for ispin
        '''#==============    Cluster MC loop   ====================='''


        ''' #===============   Heat Bath MC loop ==================
        for imove in range(lat.size()):
            # select a random spin to flip
            ispin    = np.random.randint(lat.size())
            del_ene  = ham.compute_spin_energy(ispin)  # the energy change due to spin flip
            acc_rate = 1.0/(1.0+np.exp(beta*del_ene))  # the acceptance rate
            #print(pi_plus)
            # accept/reject
            if np.random.random() < acc_rate: # if accepted flip spin to +1
                # update observables and flip spin
                lat.flip_spin(ispin)
                num_accept += 1.0
            # end if
        # end for ispin
        tot_energy  = ham.compute_energy()
        tot_mag     = lat.magnetization()
        m_per_spin2 = (tot_mag/lat.size())**2
        #==============   Heat Bath MC loop    ====================='''


        ''' #=============== Metropolis MC loop ===================
        for imove in range(lat.size()):
            # select a random spin to flip
            ispin   = np.random.randint(lat.size())
            delta_E = ham.compute_spin_energy(ispin)  # the energy change due to spin flip
            # !!!! YOU HAVE TO CODE THE MC LOOP. done!!
            # attempt to flip spin ispin
            acc_rate      = np.exp(-delta_E*beta)
            #print('acc_rate = '+str(acc_rate))
            # accept/reject
            if np.random.rand()<acc_rate: # if accepted
                # update observables and flip spin
                lat.flip_spin(ispin)
                num_accept += 1.0
                #print('num_accept = '+str(num_accept))
            # end if
        # end for ispin
        tot_energy  = ham.compute_energy()
        tot_mag     = lat.magnetization()
        m_per_spin2 = (tot_mag/lat.size())**2
        #================  Metropolis MC loop =============='''

        print(lat)

    # end for isweep loop

    print( "Acceptance Rate: %3.5f" % (float(num_accept)/nsweep/lat.size()) )

# end def mc_loop
