from __future__ import print_function
import numpy as np
from copy import deepcopy

class PairPotentialLJ:
    
    def __init__(self,sc,sig=1,eps=1,rmin=1e-16,rmax=-1):
        self._sig  = sig
        self._eps  = eps
        self._rmin = rmin
        self._rmax = rmax
        if sig < 10.*rmin:
            raise ValueError("Potential cutoff %1.2f needs to be much smaller than hard-core radius %1.2f"%(rmin,sig))
        # end if sig
        if rmax == -1:
            self._rmax = np.sqrt(3)/2.*sc.cube_length()
        # end if
        if self._rmax < sig:
            raise ValueError("Force cutoff within hard-core of LJ potential.\n\
Is the simulation box too small? rmax = %f cube_length = %f" % (rmax,sc.cube_length()) )
        # end if
        self._pset = sc.pset() # should be a reference
        self._sc   = sc
        
        # allocate arrays in constructor once
        dist_table_ref = sc.dist_table(copy=False)
        disp_table_ref = sc.disp_table(copy=False)
        
        self._force_mags= np.zeros( dist_table_ref.flatten().shape )
        self._force_mat = np.zeros( disp_table_ref.shape )
        self._tmp_accel = self._pset.all_accel(copy=True)
        self.vec_pot = np.vectorize(self.pot,otypes=[np.float])
        self.vec_force_by_r = np.vectorize(self.force_by_r,otypes=[np.float])
        
    # end def __init__

    def sig(self):
        return self._sig
    # end def

    def eps(self):
        return self._eps
    # end def
    
    def pot(self,r):
        """ evaluate pair potential at a distance r """
        # artificial cutoff to avoid blow up when applied to dist_table with a bunch of zeros
        #  if actual distances get this small, then the code will be wrong
        #  guarded by `if sig < 10.*rmin` in constructor
        if r < self._rmin:
            return 0.0 
        # end if
        return 4*self._eps*((self._sig/r)**12.-(self._sig/r)**6)
    # end def pot
    
    def force_by_r(self,r):
        """ evaluate ratio of force to distance a distance r """
        # artificial cutoff to avoid numerical issue, should be fine
        if r < self._rmin:
            return self.force_by_r(self._rmin)
        # end if
        # divide magnitude of force by r for easy use with displacement
        return 24*self._eps/r*( 2*(self._sig/r)**12.-(self._sig/r)**6 )/r
    # end def
    
    def new_accel(self):
        """ calculate acceleration of all particles in self._pset, which should be a ParticleSet, using displacement information in self._sc, which should be a SimulationCube. """

        # pass by reference because I promise to not change dist_table!
        dist_table_ref = self._sc.dist_table(copy=False)
        disp_table_ref = self._sc.disp_table(copy=False)
        
        flat_dtable = dist_table_ref.flatten()
        
        # initialize matrix of forces then flatten for easy operations
        self._force_mags[:]   = np.zeros(dist_table_ref.shape).flatten()
        idx = np.where((flat_dtable>self._rmin) & (flat_dtable<self._rmax))
        self._force_mags[idx] = self.vec_force_by_r(flat_dtable[idx])
        
        for idim in range(self._pset.ndim()):
            self._force_mat[:,:,idim] = disp_table_ref[:,:,idim] * self._force_mags.reshape(dist_table_ref.shape)
        # end for

        # satisfy Newton's 3rd law
        self._tmp_accel[:,:] = np.sum(self._force_mat - self._force_mat.transpose([1,0,2]),axis=1)
        self._tmp_accel[:,:] /= self._pset.mass()
        
        return self._tmp_accel
    # end def

# end class PairPotentialLJ

if __name__ == '__main__':

    import particleset, simulationcube
    pset = particleset.ParticleSet(2,48.0)
    cube_length = 3.0
    sc   = simulationcube.SimulationCube(cube_length,pset)
    pp   = PairPotentialLJ(sc)

    pset.change_pos(0,np.array([0.0,0.0,0.5]))
    pset.initialized = True

    dtable = sc.dist_table(copy=False)

    npoints = 20
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,0.0,myz]))
        sc.update_displacement_table()
        accel = pp.new_accel()
        print( myz,dtable[0,1],accel[1,2],accel[0,2] )
    # end for
    
# end __main__
