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
        ham = ising_hamiltonian.IsingHamiltonian(isingJ,lat)
        #print('num_accept = '+str(num_accept))
    # end if
# end for ispin

#update energy and M2
#ham = ising_hamiltonian.IsingHamiltonian(isingJ,lat)
