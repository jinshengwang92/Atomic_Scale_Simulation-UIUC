import os
from mc_ising_single import mc_loop

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Monte Carlo simulation of the Ising model with a 20x20 lattice of classical spins with magnitude 1 interacting via an Ising hamiltonian with J=1.0 at inverse temperature beta=0.25.')
    parser.add_argument('-b','--beta',type=float,default=0.3,help='inverse temperature, reasonable range 0.25-1.0')
    parser.add_argument('-n','--nsweep',type=int,default=1000,help='number of Monte Carlo sweeps')
    parser.add_argument('-sps','--spins_per_side',type=int,default=20,help='number of spins on each dimension of the cubic Ising lattice')
    parser.add_argument('-s','--seed',type=int,default=10,help='random number generator seed')
    parser.add_argument('-r','--restart',action='store_true',help='restart simulation from trajectory file, default to traj.dat')
    parser.add_argument('-j','--isingJ',type=float,default=1.0,help='Ising model\'s J parameter')
    parser.add_argument('-mag','--spin_mag',type=float,default=1.0,help='magnitude of the spins')
    args = parser.parse_args()

    # hard-coded inputs
    # ==============================
    ndump       = 25
    empty_files = True
    traj_file   = 'cluster_0.3_1000_traj.dat'
    scalar_file = 'clsuter_0.3_1000_scalar.txt'

    # parse inputs
    # ==============================
    spins_per_side = args.spins_per_side
    beta   = args.beta
    nsweep = args.nsweep
    restart= args.restart
    seed   = args.seed
    isingJ = args.isingJ
    spin_mag = args.spin_mag

    # check input & output locations
    # ==============================
    if restart: # restart from trajfile
        # check trajectory file
        if not os.path.exists(traj_file):
            print( "WARNING: no trajectory file found. Starting from scratch" )
            restart = False
        # end if
    # end if restart

    if not restart: # from scratch
        if not empty_files: # stop if any output is about to be overwritten
            for output in [traj_file,scalar_file]:
                if os.path.exists(output):
                    raise RuntimeError("%s exists!" % output)
                # end if
            # end for
        # end if

        # destroy!
        if os.path.exists(traj_file):
            open(traj_file,'w').close()
        # end if
    # end if

    # start MC
    # ==============================
    mc_loop(
        spins_per_side = spins_per_side,
        temperature    = 1.0/beta,
        nsweep         = nsweep,
        ndump          = ndump,
        restart        = restart,
        seed           = seed,
        traj_file      = traj_file,
        scalar_file    = scalar_file,
        isingJ         = isingJ,
        spin_mag       = spin_mag
    )
    #print(beta)

# end __main__
