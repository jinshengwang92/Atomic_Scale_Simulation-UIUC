import numpy as np
from copy import deepcopy

class PairPotentialLJ:
    
    def __init__(self,sc,sig=1,eps=1,rmin=1e-16,rmax=-1):
        self._sig  = sig
        self._eps  = eps
        self._rmin = rmin
        self._rmax = rmax
        if rmax == -1:
            self._rmax = np.sqrt(3)/2.*sc.cube_length()
        # end if
        self._pset = sc.pset() # should be a reference
        self._sc   = sc
        
        # allocate arrays in constructor once
        dist_table_ref = sc.dist_table(copy=False)
        disp_table_ref = sc.disp_table(copy=False)
        
        self._force_mags= np.zeros( dist_table_ref.flatten().shape )
        self._force_mat = np.zeros( disp_table_ref.shape )
        self._tmp_accel = self._pset.all_accel(copy=True)
        
    # end def __init__
    
    def pot(self,r):
        return 4*self._eps*((self._sig/r)**12.-(self._sig/r)**6)
    # end def pot
    
    def force_by_r(self,r):
        if r < self._rmin:
            return self.force_by_r(self._rmin)
        # end if
        # divide magnitude of force by r for easy use with displacement
        return 24*self._eps/r*( 2*(self._sig/r)**12.-(self._sig/r)**6 )/r
    # end def
    
    def new_accel(self):

        # pass by reference because I promise to not change dist_table!
        dist_table_ref = self._sc.dist_table(copy=False)
        disp_table_ref = self._sc.disp_table(copy=False)
        
        flat_dtable = dist_table_ref.flatten()
        
        # initialize matrix of forces then flatten for easy operations
        self._force_mags[:]   = np.zeros(dist_table_ref.shape).flatten()[:] 
        idx = np.where((flat_dtable>self._rmin) & (flat_dtable<self._rmax))
        self._force_mags[idx] = map(self.force_by_r,flat_dtable[idx]) #np.apply_along_axis(self.force_by_r,0,flat_dtable[idx])
        
        for idim in range(self._pset.ndim()):
            self._force_mat[:,:,idim] = disp_table_ref[:,:,idim] * self._force_mags.reshape(dist_table_ref.shape)
        # end for
        self._tmp_accel[:,:] = np.apply_along_axis(np.sum,1
            ,self._force_mat - self._force_mat.transpose([1,0,2]))

        '''

        self._tmp_accel[:,:] = 0.0
        for iat in range(self._pset.size()):
            for jat in range(self._pset.size()):
                if iat==jat:
                    continue
                # end if

                myiat, myjat = iat, jat
                sign = 1
                if iat > jat:
                    myiat, myjat = jat, iat
                    sign = -1
                # end if
                myr = dist_table_ref[myiat,myjat]

                self._tmp_accel[iat] += sign*self.force_by_r(
                    myr) * disp_table_ref[myiat,myjat]

            # end for jat
        # end for iat
 
        '''
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
        print myz,dtable[0,1],accel[1,2],accel[0,2]
    # end for
    
# end __main__
