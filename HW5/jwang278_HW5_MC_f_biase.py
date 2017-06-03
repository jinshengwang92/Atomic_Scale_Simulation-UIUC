#-*- encoding:utf-8 -*-
from __future__ import print_function
from particleset import ParticleSet
import numpy as np
import random
from numpy import shape
from numpy import linalg as LA
import matplotlib.pyplot as plt
import itertools
import time
import csv

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 5: Monte Carlo simulations
# ------------------------------------------------------------------------

"""
The ParticleSet class is designed to hold the position, velocity and
accelerations of a set of particles. Initialization methods
init_pos_cubic(cube_length) and init_vel(temperature) are provided
for your convenience.

pset = ParticleSet(natom) will initialize a particle set pset
pset.size() will return the number of particles in pset

| -------------------------------------------------------| ------------------- |
|                   for access of                        |    use method       |
| -------------------------------------------------------| ------------------- |
| all particle positions in an array of shape(natom,ndim)|    pset.all_pos()   |
| all particle velocities                                |    pset.all_vel()   |
| all particle accelerations                             |    pset.all_accel() |
| particle i position in an array of shape (ndim)        |    pset.pos(i)      |
| particle i velocity                                    |    pset.vel(i)      |
| particle i acceleration                                |    pset.accel(i)    |
| -------------------------------------------------------| ------------------- |

| ----------------------------- | ------------------------------------ |
|           to change           |             use method               |
| ----------------------------- | ------------------------------------ |
| all particle positions        |  pset.change_all_pos(new_pos_array)  |
| particle i position           |  pset.change_pos(i,new_pos)          |
| ditto for vel and accel       |  pset.change_*(i,new_*)              |
| ----------------------------- | ------------------------------------ |
"""

'''----------- def of pair correlation and plot the profile ---------'''

#calculate the L-J potential between iat and jat particles
def LJ_Potential(iat,jat,pset,box_length):
    dist = distance(iat,jat,pset,box_length)
    ri   = 1.0/dist
    r6i  = ri*ri*ri*ri*ri*ri
    pot  = 4.0*r6i*(r6i-1)
    return pot
#end def of L-J potential

def displacement(iat, jat, pset, box_length):
    """ Return the displacement of the iat th particle relative to the jat th
    particle. Unlike the distance function, here you will return a vector
    instead of a scalar """
    posi = pset.pos(iat)
    posj = pset.pos(jat)
    disp = posi.copy()
    idim = len(posi)      #ndim = pset.shape[0]
    for i in range(idim):
        disp[i]  = disp[i] - posj[i]
        disp[i] -= box_length*int(round(disp[i]/box_length))
        assert -box_length<disp[i]<box_length
        if disp[i]>0.5*box_length:
            disp[i] -= box_length
        if disp[i]<-0.5*box_length:
            disp[i] += box_length
    # calculate displacement of the iat particle relative to the jatth particle
    # i.e. r_i - r_j
    # be careful about minimum image convention! i.e. periodic boundary
    return disp
# end def distance

def distance(iat, jat, pset, box_length):
    """ return the distance between particle i and j according to the minimum
    image convention. """
    dist = 0.0
    vector = displacement(iat, jat, pset, box_length)
    dist = np.linalg.norm(vector)
    # calculate distance with minimum image convention
    # np.linalg.norm() may be useful here
    return dist
# end def distance


def Pair_corr(pset,box_length,num_atoms,r_num):
    '''how many steps I want to cut g(r) into, can be set in main function if
    wanted,r_num , default value is 100 here'''
    rstep = box_length/r_num/2
    rho0 = num_atoms/(box_length**3)
    g_r = np.zeros(r_num)
    d_n = np.zeros(r_num+1)
    r_i = np.zeros(r_num)
    for iat in range(num_atoms):
        for jat in range(num_atoms):
            if iat != jat:
                dist = distance(iat,jat,pset,box_length)
                if dist <= 0.5*box_length:
                    n = int(dist/rstep)  #int((dis - dis%rstep)*1.0/rstep)
                    d_n[n] += 1

    #print(d_n)
    for i in range(r_num):
        r_i[i] = (i+0.5)*rstep
        g_r[i] = d_n[i]/rho0/4/(np.pi)/rstep/r_i[i]/r_i[i]/num_atoms
    #print(g_r)
    #end for calculation
    return g_r
#end def Pair_corr

def plot_pair_corr(r_num,g_r_ave,box_length,temperature,nsteps,sigma):
    rstep = box_length/r_num/2
    r_i = np.zeros(r_num)
    for i in range(r_num):
        r_i[i] = (i+0.5)*rstep
    #plot for g(r)
    plt.figure('Pair_corr')
    plt.plot(r_i,g_r_ave,label='pair correlation g(r)')
    plt.plot([0, box_length/2], [1, 1], '--')
    plt.scatter(r_i,g_r_ave)
    plt.legend()
    plt.xlim([0,box_length/2+0.1])
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.grid()
    #plt.ylim([-0.5,4])  # range of y axiis to compare between figures
    plt.title('Plot of pair correlation g(r) at T = %.3f'%temperature)
    plt.savefig('Pair_corr_T_%.3f_%d_%.3f.png'%(temperature,nsteps,sigma))
    plt.show()

#end for plot of figure

'''----------------------- end def of pair correlation ----------------------'''


'''----------functions for calculation for structure factors----------------'''
#def of computaiton for legal k-vectors
def legal_kvecs(maxk,ndim,box_length):
    #maximal n is maxk
    kvec = []
    kvec = np.array([np.array(i) for i in itertools.product(range(maxk+1),
    repeat=ndim)])
    kvec_tem = np.zeros((len(kvec)-1,3))
    for i in range(len(kvec)-1):
        kvec_tem[i]=kvec[i+1]
    kvecs = 2*np.pi/box_length*kvec_tem
    #print(kvecs)
    return kvecs
#end def of legal k-vectors

#def of rhok
def rhok(kvec,pset):
    value = 0.0+0.0*1j
    natom = pset.size()
    pos = pset.all_pos()
    for i in range(natom):
        x = np.dot(kvec,pos[i])
        value += np.cos(x) - np.sin(x)*1j
    #compute \sum_j \exp(i*k\dot r_j)
    return value
#end def of rhok

#def of calculation of all the structure factors s(k)
def Sk(kvecs,pset):
    length = len(kvecs)
    natom = pset.size()
    sk_list = np.zeros(length)
    for i in range(length):
        rho = rhok(kvecs[i],pset)
        rho_sq = rho.real*rho.real + rho.imag*rho.imag
        sk_list[i] = rho_sq/natom
    return sk_list
#end def of Sk

#def how many points for s(k) data
def num_k_mag(maxkk,ndim):
    kvecs = np.array([np.array(i) for i in itertools.product(range(maxkk+1),
    repeat=ndim)])
    kmags  = [LA.norm(kvec) for kvec in kvecs]
    unique_kvecs = np.unique(kmags)
    num = len(unique_kvecs)+1
    return num
#end define the num of data points for s(K)

#calculate for struture factors
def structure_r(pset,maxk,box_length):
    ndim    = pset.ndim()
    kvecs   = legal_kvecs(maxk,ndim,box_length)
    sk_list = Sk(kvecs,pset)
    #average sk and plot the final results
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    sk_arr = np.array(sk_list)   # convert to numpy array type
    #average s(k) if different k-vectors share the same magnitude
    unique_kmags = np.unique(kmags)
    unique_sk    = np.zeros(len(unique_kmags))
    for iukmag in range(len(unique_kmags)):
        kmag    = unique_kmags[iukmag]
        idx2avg = np.where(kmags == kmag)
        unique_sk[iukmag] = np.mean(sk_arr[idx2avg])
    #end for loop of iukmag
    return unique_sk
#end def structure_r

#plot structure_factors
def plot_str_factor(s_r_ave,maxk,box_length,temperature,ndim,nsteps,sigma):
    kvecs   = legal_kvecs(maxk,ndim,box_length)
    #average sk and plot the final results
    kmags  = [np.linalg.norm(kvec) for kvec in kvecs]
    #average s(k) if different k-vectors share the same magnitude
    unique_kmags = np.unique(kmags)
    #visualize for s(k)
    plt.figure('structure_factors')
    plt.plot(unique_kmags,s_r_ave,label='structure factors s(k)')
    plt.scatter(unique_kmags,s_r_ave)
    plt.legend()
    plt.xlabel('k magnitude')
    plt.ylabel('s(k) magnitude')
    plt.grid()
    #plt.ylim([-1,20])  # ylimt to better compare figures
    plt.title('s(k) when kmax = %d and T = %.3f'%(maxk,temperature))
    plt.savefig('Str_fac_T_%.3f_%d_%.3f.png'%(temperature,nsteps,sigma))
    plt.show()
#end def of plot_str_factor
'''---------end of calculation for structure factors-----------'''

#de for calcualting potential for given pset
def compute_potential(pset,box_length):
    natom         = pset.size()  # number of particles
    #mass          = pset.mass()
    #ndim          = pset.ndim()
    tot_potential = 0.0
    #loop for potential
    for iat in range(natom):
        for jat in range(iat):
            tot_potential += LJ_Potential(iat,jat,pset,box_length)
    #end loop
    return tot_potential
#end def

#def for calculating potential related to iat atoms
def compute_iat_potential(iat,pset,box_length):
    natom         = pset.size()  # number of particles
    tot_iat_pot   = 0.0
    for jat in range(natom):
        if jat != iat:  # not calculating with itself
            tot_iat_pot += LJ_Potential(iat,jat,pset,box_length)
        #end if
    #end for
    return tot_iat_pot
#end def


def compute_iat_force(iat,pset,box_length):
    """
    We want to return the force on atom 'iat' when we are given a list of
    all atom positions. Note, pos is the position vector of all the
    atom and pos[0][0] is the x coordinate of the 1st atom. It may
    be convenient to use the 'displacement' function above. For example,
    disp = displacement( 0, 1, pset, box_length ) would give the position
    of the 1st atom relative to the 2nd, and disp[0] would then be the x
    coordinate of this displacement. Use the Lennard-Jones pair interaction.
    Be sure to avoid computing the force of an atom on itself.
    """

    natoms = pset.size()
    dim    = pset.ndim()
    force  = np.zeros(dim)  # allocate the space for force
    for jat in range(natoms):
        if jat != iat:
            disp = displacement(iat,jat,pset,box_length)  #displacement i to j
            dist = distance(iat, jat, pset, box_length)
            if dist <= 0.5*box_length:  #force cutoff is half of the box_length
                ri  = 1.0/dist
                r6i = ri**6
                rforce = 24*r6i*ri*(2*r6i-1)   # radial force on i

                #force in 3 cardisian coordinates
                for i in range(dim):
                    force[i] += rforce*disp[i]*ri
                #end for
            #end if

        else:
            continue
    # calculate force
    return force
# end def internal_force

#plot for energy
def plot_potential(pot_all,nsteps,temperature,sigma):
    #plot for figure TE,PE,KE
    plt.figure('plot_of_potential_energy')
    x_axis = [i for i in range(nsteps)]
    plt.plot(x_axis,pot_all,label='Potential Energy')
    plt.legend()
    plt.xlabel('MC sweeps')
    plt.ylabel('Potential Energy')
    plt.title(r'potential energy when sigma=%.3f, %d sweeps'%(sigma,nsteps))
    plt.savefig('pot_%.3f_%d_%.3f.png'%(temperature,nsteps,sigma))
    plt.show()
#end for plot of figure


#def of updata_position by force biased
def update_position_force(pset,iat,sigma,force,mass,dt):
    # should return new position particle iat in pset
    # using the Gaussian distribution described above and force biased
    old_ptcl_pos = pset.pos(iat)
    eta_x        = np.random.normal(0.0,sigma,3)   # eta_x sampled
    x_adjust     = force/2.0/mass*dt*dt
    new_ptcl_pos = old_ptcl_pos + eta_x + x_adjust
    return new_ptcl_pos, eta_x
# end def update_position


if __name__ == '__main__':
    start_time = time.time()

    num_atoms   = 64
    mass        = 48.0
    dt          = 0.032      # time step 0.032
    temperature = 2.0   # initla temperature 0.728
    #Tem_bath    = 0.1   # the temperature of the heat bath
    #possibility = 0.5   # the possibility of collision
    #eta         = possibility/dt  #coupling strength
    box_length  = 4.0 # side length of the box, decided by density = 1,
                            #=4.2323167 previously
    nsteps      = 3000       # total step for simulation
    accept      = 0.0
    reject      = 0.0
    momen       = np.zeros((nsteps,3))  #used to store momentum
    pot_all     = np.zeros(nsteps)  # store energy
    pos_par_0   = np.zeros((nsteps,3))  # the postions of particle 0
    veloc_t0    = np.zeros((num_atoms,3))  #init velocity for all the particles
    veloc_t     = np.zeros((num_atoms,3))  # store vel. for particles at t
    #vv_a        = np.zeros(nsteps)      # store velocity products at tj steps
    #Inst_T_all  = np.zeros(nsteps)      # store instananeous time
    seed        = 3   # change seed to initial velocity
    r_num       = 50 # the pieces of g(r) is cut into
    maxk        = 5   #max value of k vector component
    firstcutoff = 1000  # the start of equillibrium
    g_r_sum     = np.zeros(r_num)
    s_r_sum     = np.zeros(num_k_mag(maxk,3))
    sigma       = 0.04   # the std dev for position update
    beta        = 1.0/temperature
    poten       = []  #store pot. after equillibrium

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    #pset._alloc_pos_vel_accel()
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature,seed)

    # Monte Carlo simulation main loop

    for istep in range(nsteps):

        old_pot = compute_potential(pset,box_length)
        pot_all[istep] = old_pot
        print('*'*60)
        print(istep,pot_all[istep])
        # update positions

        for iat in range(num_atoms):
            old_pos = pset.pos(iat)   # save old position in case of rejection
            old_iat_pot = compute_iat_potential(iat,pset,box_length)
            old_iat_force = compute_iat_force(iat,pset,box_length)
            new_pos = update_position_force(pset,iat,sigma,old_iat_force,mass,dt)[0]
            # new position values returned first
            ETA     = update_position_force(pset,iat,sigma,old_iat_force,mass,dt)[1]
            #eta values do not have to sample again
            pset.change_pos(iat,new_pos)
            new_iat_pot = compute_iat_potential(iat,pset,box_length)
            new_iat_force = compute_iat_force(iat,pset,box_length)
            delta_pot   = new_iat_pot - old_iat_pot
            prop_ratio  = ((LA.norm(ETA+old_iat_force/2.0/mass*dt*dt))**2 -
            (LA.norm(ETA+new_iat_force/2.0/mass*dt*dt))**2)/2.0/sigma/sigma
            # propose ratio
            A_old_new = min(1.0, np.exp(-beta*delta_pot+prop_ratio))
            #print('Acceptance A = '+ str(A_old_new))
            if A_old_new < np.random.random():
                reject += 1.0
                pset.change_pos(iat,old_pos)    # return to the old position
            #end if
        # end for iat

        if istep >=firstcutoff:  # take the average value of g(r) and s(r)
            g_r_sum += Pair_corr(pset,box_length,num_atoms,r_num)
            s_r_sum += structure_r(pset,maxk,box_length)
            poten.append(old_pot)
        #end if

    #end for loop

    #print out reject rate
    reject_rate = reject/num_atoms/nsteps
    accept_rate = 1.0 - reject_rate
    print('*'*60)
    print('accept rate = '+str(accept_rate)+' when sigma = '+str(sigma))
    print('*'*60)

    # do pair correlation calculation and plot
    g_r_ave = 1.0/(nsteps-firstcutoff)*g_r_sum
    plot_pair_corr(r_num,g_r_ave,box_length,temperature,nsteps,sigma)

    #----------calculation and plot for structure factors s(k) -----------
    s_r_ave = 1.0/(nsteps-firstcutoff)*s_r_sum
    plot_str_factor(s_r_ave,maxk,box_length,temperature,3,nsteps,sigma)

    #-------------------------  plot for energy  -------------------------
    plot_potential(pot_all,nsteps,temperature,sigma)
    
    #print out how long the code run for
    print("--- the code runs %.3f seconds ---" % (time.time() - start_time))

    #write the equillibrium poten to a file for later calcualtion
    p_template = 'pot_{0}_{1}_{2}.txt'
    filename   = p_template.format(temperature,nsteps,sigma)
    with open(filename,'w') as fin:
        writer = csv.writer(fin,lineterminator='\n')
        for val in poten:
            writer.writerow([val])
    #end write to the file

# end __main__
#
