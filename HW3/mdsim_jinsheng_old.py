#-*- encoding:utf-8 -*-
from __future__ import print_function
from particleset import ParticleSet
import numpy as np
import random
from numpy import shape
import matplotlib.pyplot as plt
import itertools

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 3: Introduction to Molecular Dynamics
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

# Routines to ensure periodic boundary conditions that YOU must write.
# ------------------------------------------------------------------------
def pos_in_box(mypos, box_length):
    """ Return position mypos in simulation box using periodic boundary
    conditions. The simulation box is a cube of size box_length^ndim centered
    around the origin vec{0}. """

    new_pos = mypos.copy()
    ndim    = mypos.shape[0]
    for i in range(ndim):
        new_pos[i] -= box_length*int(round(new_pos[i]/box_length))
    # calculate new_pos
    return new_pos
# def pos_in_box

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

# The Verlet time-stepping algorithm that YOU must write, dt is time step
# ------------------------------------------------------------------------
def verlet_next_pos(pos_t,vel_t,accel_t,dt):
    """
    We want to return position of the particle at the next moment t_plus_dt
    based on its position, velocity and acceleration at time t.
    """
    pos_t_plus_dt = pos_t.copy()
    ndim = len(pos_t)
    for i in range(ndim):
        pos_t_plus_dt[i] += dt*(vel_t[i]+0.5*dt*accel_t[i])
    #end for
    return pos_t_plus_dt
# end def verlet_next_pos

def verlet_next_vel(vel_t,accel_t,accel_t_plus_dt,dt):
    """
    We want to return velocity of the particle at the next moment t_plus_dt,
    based on its velocity at time t, its acceleration at time t and t_plus_dt
    """
    vel_t_plus_dt = vel_t.copy()
    ndim = len(vel_t)
    for i in range(ndim):
        vel_t_plus_dt[i] += 0.5*dt*(accel_t[i] + accel_t_plus_dt[i])
    #end for
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

#calculate the L-J potential between iat and jat particles
def LJ_Potential(iat,jat,pset,box_length):
    dist = distance(iat,jat,pset,box_length)
    ri   = 1.0/dist
    r6i  = ri**6
    pot  = 4.0*r6i*(r6i-1)
    return pot
#end def of L-J potential

def compute_energy(pset,box_length):
    natom = pset.size()  # number of particles
    mass  = pset.mass()
    vel   = pset.all_vel() # all particle velocies
    ndim  = pset.ndim()


    tot_kinetic   = 0.0    #0.5*mass*(vel**2).sum()
    tot_potential = 0.0
    for iat in range(natom):
        for idim in range(ndim):
            tot_kinetic += vel[iat][idim]*vel[iat][idim]
    tot_kinetic = 0.5*mass*tot_kinetic
    #end cal for total kenetic energy

    for iat in range(natom):
        for jat in range(iat):
            if distance(iat,jat,pset,box_length) <= 0.5*box_length:
                # set potential cutoff at r=0.5*box_length
                tot_potential += LJ_Potential(iat,jat,pset,box_length)
            else:
                continue
    #end total potential energy calcualtion

    tot_energy = tot_kinetic + tot_potential
    return (tot_kinetic, tot_potential, tot_energy)
#end of the def of compute_energy

#plot for energy
def plot_energy(energy_all,nsteps,dt):
    #plot for figure TE,PE,KE
    plt.figure('plot_of_energy')
    x_axis = [dt*i for i in range(nsteps)]
    y_KE = energy_all[:,0]      #kenetic energy
    y_PE = energy_all[:,1]      #potential energy
    y_TE = energy_all[:,2]      #total energy
    plt.plot(x_axis,y_KE,label='Kinetic Energy')
    plt.plot(x_axis,y_PE,label='Potential Energy')
    plt.plot(x_axis,y_TE,label='Total Energy')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Energy')
    plt.title(r'Plot of energy when dt=%.2f, %d steps'%(dt,nsteps))
    plt.show()
#end for plot of figure


''' ---------- definiton of the instantaneous temperature -----------'''
def instant_T (total_kenetic_energy,num_atoms):
    #input total kenetic energy and num of atoms
    tem = total_kenetic_energy*2/3/num_atoms    #calcualtion of the temperature
    return tem                  #return the value of instananeous temperature
# end of defintion of instananeous temperature

'''---------------- def of momentum of the system -----------------'''
def momentum(velocity_all,mass):
    mom = [0,0,0]
    for i in range(3):
        for vel in velocity_all:
            mom[i] += mass*vel[i]
    #end for three dimensions
    return mom
#end of def of momentum

'''----------- def of pair correlation and plot the profile ---------'''
def Pair_corr(pset,box_length,num_atoms,r_num):
    '''how many steps I want to cut g(r) into, can be set in main function if
    wanted,r_num , default value is 100 here'''
    rstep = box_length/r_num/2
    rho0 = num_atoms/(box_length**3)
    g_r = np.zeros(r_num)
    d_n = np.zeros(r_num)
    r_i = np.zeros(r_num)
    for i in range(num_atoms):
        for j in range(i+1,num_atoms):
            dis = distance(i,j,pset,box_length)
            if dis <= box_length*1.0/2:
                n = int(dis*1.0/rstep)  #int((dis - dis%rstep)*1.0/rstep)
                d_n[n] += 2
            else:
                continue
    print(d_n)
    for i in range(r_num):
        r_i[i] = (i+0.5)*rstep
        g_r[i] = d_n[i]*1.0/rho0/4/(np.pi)/rstep/r_i[i]/r_i[i]/num_atoms
    #print(g_r)
    #end for calculation
    return g_r
#end def Pair_corr

def plot_pair_corr(r_num,g_r_ave,box_length):
    rstep = box_length/r_num/2
    r_i = np.zeros(r_num)
    for i in range(r_num):
        r_i[i] = (i+0.5)*rstep
    #plot for g(r)
    plt.figure('Pair_corr')
    plt.plot(r_i,g_r_ave,label='pair correlation g(r)')
    plt.legend()
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.title(r'Plot of pair correlation fucntion g(r)')
    plt.show()
#end for plot of figure

'''------------------------ end def of pair dis -----------------------------'''


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

#plot structure_factors
def plot_str_factor(pset,maxk,box_length):
    ndim = pset.ndim()
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
    #visualize for s(k)
    plt.figure('structure_factors')
    plt.plot(unique_kmags,unique_sk,label='structure factors s(k)')
    plt.legend()
    plt.xlabel('k magnitude')
    plt.ylabel('s(k) magnitude')
    plt.title('s(k) for different k when kmax = %d'%maxk)
    plt.show()
#end def of plot_str_factor
'''---------end of calculation for structure factors-----------'''


'''---------calculation for diffusion constant D given all the vv in array---'''
#def of diffusion and plot of vv
def D_vv_plot(vv_a,dt):
    DIFF = np.sum(vv_a)*dt
    print('Diffusion constant = %f'%DIFF)
    plt.figure('velocity-velocity correlation vs time')
    t = [dt*step for step in range(nsteps)]
    plt.plot(t,vv_a,label='velocity-velocity correlation')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('v-v correlation')
    plt.title('velocity-velocity correlation')
    plt.show()
#end def of D_vv_plot

#def of plot of insant temperature
def plot_instant_T(dt,nsteps,Inst_T_all):
    x = [dt*i for i in range(nsteps)]
    plt.figure('Instaneous temperature vs time')
    plt.plot(x,Inst_T_all,label='Instaneous temperature')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('Instaneous temperature')
    plt.title('Instaneous temperature vs time')
    plt.show()

#end of def of plot of instant_T
# end def compute_energy


if __name__ == '__main__':

    num_atoms   = 64
    mass        = 48.0
    dt          = 0.01      # time step
    temperature = 0.728    # initla temperature
    Tem_bath    = 0.728   # the temperature of the heat bath
    possibility = 0.01     # the possibility of collision
    eta         = possibility/dt  #coupling strength
    box_length  = 4.2323167 # side length of the box, decided by density = 1,
                            #=4.2323167 previously
    nsteps      = 100       # total step for simulation
    energy_all  = np.zeros((nsteps,3))  # store energy
    pos_par_0   = np.zeros((nsteps,3))  # the postions of particle 0
    veloc_t0    = np.zeros((num_atoms,3))  #init velocity for all the particles
    veloc_t     = np.zeros((num_atoms,3))  # store vel. for particles at t
    vv_a        = np.zeros(nsteps)      # store velocity products at tj steps
    Inst_T_all  = np.zeros(nsteps)      # store instananeous time
    seed        = 2   # change seed to initial velocity
    r_num       = 200 # the pieces of g(r) is cut into
    maxk        = 4   #max value of k vector component
    firstcutoff  = 80  # the start of equillibrium
    g_r_ave     = np.zeros(r_num)

    # create and initialize a set of particles
    pset = ParticleSet(num_atoms,mass)
    #pset._alloc_pos_vel_accel()
    pset.init_pos_cubic(box_length)
    pset.init_vel(temperature,seed)

    # molecular dynamics simulation loop

    for istep in range(nsteps):

        # calculate properties of the particles
        Ene_all = compute_energy(pset,box_length)
        print(istep, Ene_all[0],Ene_all[1],Ene_all[2])

        #record the initial velocity for all the particles
        if istep == 0:
            veloc_t0 = pset.all_vel()
        #end if

        veloc_t = pset.all_vel()   # current velocities of all particles
        # get all the values of ave_vv for all the steps
        su_jj = 0.0
        for i in range(num_atoms):
            su_jj += np.dot(veloc_t0[i],veloc_t[i])
        #end for
        vv_a[istep] = su_jj/num_atoms
        vv_a[istep] = vv_a[istep]/vv_a[0]   #normalization vv by v(0)v(0)
        #get and print the instananeous temperature for every step
        ins_T             = instant_T(Ene_all[0],num_atoms)
        Inst_T_all[istep] = ins_T
        #print('instantaneous temperature = %f'%ins_T)

        #get the momentum
        momen = momentum(pset.all_vel(),mass)
        #print('current momentum = '+ str(momen))
        #print(momen)
        #print(np.sqrt(momen[0]**2+momen[1]**2+momen[2]**2))

        #put all the energy together for plot of energy curve
        energy_all[istep][0] = Ene_all[0]  #kenetic energy
        energy_all[istep][1] = Ene_all[1]  #potential energy
        energy_all[istep][2] = Ene_all[2]  #total energy

        # update positions
        for iat in range(num_atoms):

            old_acc = internal_force(iat,pset,box_length)/mass
            pset.change_accel(iat,old_acc)
            my_next_pos = verlet_next_pos( pset.pos(iat),pset.vel(iat), pset.accel(iat), dt)
            new_pos = pos_in_box(my_next_pos,box_length)
            if iat == 0:        # find and store the position for particle 0
                pos_par_0[istep,:] = new_pos
            pset.change_pos(iat,new_pos)
            my_new_acc = internal_force(iat,pset,box_length)/mass
            pset.change_accel(iat,my_new_acc)
            my_next_vel = verlet_next_vel(pset.vel(iat), old_acc,pset.accel(iat), dt )
            pset.change_vel(iat, my_next_vel )

            #determine whether have to change velocity by Anderson Thermostat
            '''
            if random.random() > possibility:
                v1 = random.gauss(0,np.sqrt(Tem_bath/mass))
                v2 = random.gauss(0,np.sqrt(Tem_bath/mass))
                v3 = random.gauss(0,np.sqrt(Tem_bath/mass))
                pset.change_vel(iat, np.array([v1,v2,v3]))
            #end if
            '''
        #end for iat

        if istep >firstcutoff:  # take the average value of g(r)
            g_r_ave += 1.0/(nsteps-firstcutoff)*Pair_corr(pset,box_length,num_atoms,r_num)
        #end if
    #end for loop

    # to get the postions of the particel 0
    #print(pos_par_0)

    # do pair correlation calculation and plot
    plot_pair_corr(r_num,g_r_ave,box_length)


    #----------calculation and plot for structure factors s(k) -----------
    plot_str_factor(pset,maxk,box_length)


    #------ calculation, plot for v-v corre. and diff. constant D  ------
    D_vv_plot(vv_a,dt)


    #-------------------------  plot for energy  -------------------------
    plot_energy(energy_all,nsteps,dt)

    #---------------------------plot instant_T  --------------------------
    plot_instant_T(dt,nsteps,Inst_T_all)

# end __main__
#
