import numpy as np
from copy import deepcopy

class SimulationCube:
    
    def __init__(self,cube_length,pset):
        self._cube_length = cube_length
        self._cube_side   = cube_length/2.
        self._rlimit      = self._cube_side
        # largest distance to not include periodic image

        self._pset = pset # keep a REFERENCE to pset in order to track changes
        # do NOT edit internals of pset in this class!
        
        # allocate local arrays to avoid allocation at each simulation step
        self._cur_pos = pset.all_pos(copy=False) # don't change _cur_pos!
        # i.e. self._cur_pos should NEVER be the left operator of assign (=) later

        # displacement and distance tables
        self._disp_table = np.zeros([pset.size(),pset.size(),pset.ndim()])
        self._dist_table = np.zeros([pset.size(),pset.size()])
        # note: sparse matrix available in scipy can reduce memory requirement
        #  only the upper triangular part of the tables are used, i.e.
        #    table[i,j] is only non-zero if i<j
        
    # end def
    
    # ---------------------------
    # begin accessor methods

    def pset(self):
        return self._pset
    # end def
    
    def cube_length(self):
        return self._cube_length
    # end def

    def rlimit(self):
        return self._rlimit
    # end def

    def volume(self):
        return self._cube_length**3.
    # end def
    
    def disp_table(self,copy=True):
        if copy:
            return deepcopy( self._disp_table )
        else:
            return self._disp_table
        # end if
    # end def
    
    def dist_table(self,copy=True):
        if copy:
            return deepcopy( self._dist_table )
        else:
            return self._dist_table
        # end if
    # end def

    # end accessor methods
    # ---------------------------
    
    def pos_in_box(self,cur_pos):
        """ return posistions of cur_pos in simulation box, vectorized to operate on pset.all_pos() """
        
        # locate position components that need fix
        out_idx = np.where( (cur_pos >=  self._cube_side) |\
                            (cur_pos < -self._cube_side) )

        # put bad components back into simulation box
        if len(out_idx[0]) > 0:
            cur_pos[out_idx] = (cur_pos[out_idx]+self._cube_side)%(self._cube_length) - self._cube_side 
        # end if
        
        return cur_pos
    # end def pos_in_box
    
    def update_displacement_table(self):
        """ update self._disp_table and self._dist_table """
        # !!!! assume particles are already all in box
        
        # generate a list of indices for pairs of particles
        pair_list = [ (i,j) for i in range(self._pset.size()-1) for j in range(i+1,self._pset.size()) ]

        # calculate open-boundary displacements for every pair of particles
        for pair in pair_list:
            iat,jat = pair
            self._disp_table[iat,jat,:] = self._cur_pos[iat] - self._cur_pos[jat]
        # end for pair
        
        # locate displacement components that need fix
        out_idx = np.where( abs(self._disp_table) > self._cube_side )
        cur_out_disp = self._disp_table[out_idx]
        
        # fix displacement components
        cur_out_sign = np.sign(cur_out_disp)
        cur_out_leng = np.abs(cur_out_disp)
        self._disp_table[out_idx] = -cur_out_sign*(self._cube_length - cur_out_leng)

        # calculate distances from displacements
        self._dist_table = np.linalg.norm(self._disp_table,axis=2)

    # end def update_displacement_table

    def update_for_particle(self,iat):
        """ update disp_table and dist_table after a single particle iat has changed its position. """

        for jat in range(self._pset.size()):
            dvec = self._cur_pos[iat] - self._cur_pos[jat]
            out_idx = np.where( abs(dvec) > self._cube_side )
            cur_out_disp = dvec[out_idx]
            cur_out_sign = np.sign(cur_out_disp)
            cur_out_leng = np.abs(cur_out_disp)
            dvec[out_idx]= -cur_out_sign*(self._cube_length - cur_out_leng)

            if iat > jat:
                self._disp_table[jat,iat,:] = dvec[:]
                self._dist_table[jat,iat] = np.linalg.norm(dvec)
            elif iat < jat:
                self._disp_table[iat,jat,:] = dvec[:]
                self._dist_table[iat,jat] = np.linalg.norm(dvec)
            else: # iat == jat
                pass
            # end if
        # end for jat

    # end def update_for_particle

# end class SimulationCube

if __name__ == '__main__':
    # unit tests
    # remember to check displacement direction!

    import particleset

    num_atoms   = 2
    mass        = 1.0
    cube_length = 4.3

    pset = particleset.ParticleSet(num_atoms,mass)
    sc   = SimulationCube(cube_length,pset)

    npoints = 20

    # check z direction
    pset.change_pos(0,np.array([0.0,0.0,0.0]))
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,0.0,myz]))
        sc.update_displacement_table()
        dist = sc.dist_table(copy=False)[0,1]
        disp = sc.disp_table(copy=False)[0,1]
        assert dist == abs(myz)
        assert dist <= np.sqrt(3)/2*cube_length
        assert abs(disp[2])==dist
        if myz < 0:
            assert disp[2] > 0
        else:
            assert disp[2] < 0 
        # end if
    # end for myz

    pset.change_pos(0,np.array([0.0,0.0,-cube_length/2.+0.1]))
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,0.0,myz]))
        sc.update_displacement_table()
        dist = sc.dist_table(copy=False)[0,1]
        disp = sc.disp_table(copy=False)[0,1]
        if myz<0.1:
            assert np.allclose(dist, abs(myz+cube_length/2.-0.1))
        else:
            assert np.allclose(dist,cube_length/2.-myz+0.1)
        # end if
        assert dist <= np.sqrt(3)/2*cube_length
        assert abs(disp[2])==dist
        if (myz>-cube_length/2.0+0.1) and (myz<0.1):
            assert disp[2] < 0
        else:
            assert disp[2] > 0 
        # end if
    # end for

    # check y direction
    pset.change_pos(0,np.array([0.0,0.0,0.0]))
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,myz,0.0]))
        sc.update_displacement_table()
        dist = sc.dist_table(copy=False)[0,1]
        assert dist == abs(myz)
        assert dist <= np.sqrt(3)/2*cube_length
        assert abs(disp[1])==dist
    # end for myz

    pset.change_pos(0,np.array([0.0,-cube_length/2.+0.2,0.0]))
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,myz,0.0]))
        sc.update_displacement_table()
        dist = sc.dist_table(copy=False)[0,1]
        if myz<0.2:
            assert np.allclose(dist, abs(myz+cube_length/2.-0.2))
        else:
            assert np.allclose(dist, abs(cube_length/2.-myz+0.2))
        # end if
        assert dist <= np.sqrt(3)/2*cube_length
    # end for

    pset.change_pos(0,np.array([0.0,cube_length/2.-0.1,0.0]))
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,myz,0.0]))
        sc.update_displacement_table()
        dist = sc.dist_table(copy=False)[0,1]
        if myz<-0.1:
            assert np.allclose(dist, cube_length/2.+myz+0.1)
        else:
            assert np.allclose(dist, abs(myz-cube_length/2.+0.1))
        # end if
        assert dist <= np.sqrt(3)/2*cube_length
    # end for

    pset.change_pos(0,np.array([0.0,cube_length/2.-0.1,0.0]))
    sc.update_displacement_table()
    for myz in np.linspace(-cube_length/2.,cube_length/2.,npoints):
        pset.change_pos(1,np.array([0.0,myz,0.0]))
        sc.update_for_particle(1)
        dist = sc.dist_table(copy=False)[0,1]
        if myz<-0.1:
            assert np.allclose(dist, cube_length/2.+myz+0.1)
        else:
            assert np.allclose(dist, abs(myz-cube_length/2.+0.1))
        # end if
        assert dist <= np.sqrt(3)/2*cube_length
    # end for

# end __main__
