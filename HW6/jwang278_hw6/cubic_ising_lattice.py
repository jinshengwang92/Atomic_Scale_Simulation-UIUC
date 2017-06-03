import numpy as np

# ------------------------------------------------------------------------
# PHY466/MSE485 Atomic Scale Simulations
# Homework 6: Ising Lattice
# ------------------------------------------------------------------------

class CubicIsingLattice:
    """ The CubicIsingLattice class is designed to hold a lattice of classical spins on a cubic lattice. self._spins is NOT meant to be accessed or modified directly. Please use accessor (spins()) and modifier (flip_spin()) methods instead.
    Spins are considered to be labeled either by a linear index or a multi-dimensional index. For example, on a 3x3 lattice, spin (0,0) has linear index 0, spin (0,1) has linear index 1, spin (1,0) has linear index 3. You can convert a multi-dimensional index to a linear index with the multi_idx() method, and vice versa with linear_idx(). They are simple wrappers for np.ravel_multi_index() and np.unravel_index() with appropriate guards.
  The neighbor list stored in self._nb_list (accessed by nb_list()) contains, for each spin, a list of linear indices of its neighbours. The list is initialized with the neighbours() method, which is only called in the constructor.

lat = CubicIsingLattice(3) will initialize a 3x3 lattice of spins
lat.size() will return the total number of spins on the lattice

| --------------------------------------------------------- | ------------------- |
|                   for access of                           |    use method       |
| --------------------------------------------------------- | ------------------- |
| all spin states                                           |    lat.spins()      |
| magnitude of the spins (assumed to be the same)           |    lat.spin_mag()   |
| spin ispin state                                          |    lat.spin(ispin)  |
| number of spins per dimension                             |    lat.nside()      |
| shape of latice (default: [lat.nside(),lat.nside()])      |    lat.shape()      |
| neighbour list                                            |    lat.nb_list()    |
| multi-dimensional index of spin ispin                     |lat.linear_idx(ispin)|
| linear index of spin labeled by tuple index idx           | lat.multi_idx(idx)  |
| --------------------------------------------------------- | ------------------- |

| ----------------------------- | --------------------------- |
|           to change           |         use method          |
| ----------------------------- | --------------------------- |
| spin state of ispin           |    lat.flip_spin(ispin)     |
| ----------------------------- | --------------------------- |
    """

    def __init__(self,spins_per_dim,ndim=2,spin_mag=1.0):

        # save input quantities
        self._spins_per_dim = spins_per_dim # number of spins along each dimension of the cubic lattice
        self._ndim     = ndim # number of dimensions
        self._spin_mag = spin_mag # magnitude of a single spin (like mass)
        self._nb_per_dim = 2 # 2 neighbours per dimension on a cubic lattice

        # save calculated quantities
        self._shape = [spins_per_dim]*ndim # default: (spins_per_dim,spins_per_dim)
        self._nspin = spins_per_dim**ndim  # total number of spins

        # initialize all spins to be pointing up
        self._spins = 2*np.random.randint(2,size=self._shape)-1 # initialize cubic lattice of spins
        #np.ones(self._shape,dtype=int)
        #2*np.random.randint(2,size=self._shape)-1
        # yes, yes, an array of bits would have been much more memory efficient
        #  unfortunately bit array is not a built-in python object

        # allocate and initialize neighbor list to establish the topology of the lattice
        self._nb_list = np.zeros([self._nspin,self._nb_per_dim*self._ndim],dtype=int)
        for ispin in range(self._nspin): # calculate and save the neighbours of each spin
            # !!!! YOU HAVE TO IMPLEMENT "neighbours(self,ispin)"
            self._nb_list[ispin,:] = self.neighbours(ispin)
        # end for ispin

    # end def __init__

    def __str__(self):
        # allow lattice to be printed
        return str(self._spins)
    # end def

    # ---------------------------
    # begin accessor methods

    def ndim(self):
        return self._ndim
    # end def ndim

    def size(self):
        return self._nspin
    # end def size

    def nside(self):
        return self._spins_per_dim
    # end def nside

    def spin_mag(self):
        return self._spin_mag
    # end def spin_mag

    def shape(self):
        return self._shape[:] # return a copy to deter external modification
    # end def shape

    def spins(self,copy=True):
        if copy:
            return self._spins.copy()
        else:
            return self._spins
        # end if
    # end def spins

    def spin(self,ispin):
        spin_idx = self.multi_idx(ispin)
        return self._spins[spin_idx]
    # end def spin

    def nb_list(self,copy=True):
        if copy:
            return self._nb_list.copy()
        else:
            return self._nb_list
        # end if
    # end def nb_list

    def linear_idx(self,tuple_idx):
        """ locate the linear index of a spin on the lattice: this method takes a multi-dimensional index and returns a single integer that labels the selected spin """
        tuple_idx = np.array(tuple_idx)
        # guards
        out_lower_bound = np.where(tuple_idx<0)
        out_upper_bound = np.where(tuple_idx>=self.nside())
        if len(out_lower_bound[0]) != 0:
            raise IndexError(  "some components of index %s out of lower bounds" % str(tuple_idx)  )
        elif len(out_upper_bound[0]) != 0:
            raise IndexError(  "some components of index %s out of upper bounds" % str(tuple_idx)  )
        # end if

        ispin = np.ravel_multi_index(tuple_idx,self.shape())
        return ispin
    # end def linear_idx

    def multi_idx(self,ispin):
        """ locate the multi-dimensional index of a spin on the lattice: this method takes a index and returns a multi-dimentional index """
        if ispin >= self.size() or ispin<0: # guard against misuse
            raise IndexError("linear spin index %d is out of bounds."%ispin)
        # end if
        return np.unravel_index(ispin,self.shape())
    # end def multi_idx

    # end accessor methods
    # ---------------------------

    # ---------------------------
    # begin modifier methods

    def flip_spin(self,ispin):
        """ flip spin ispin and return the change in total magnetization """
        dmag = self.mag_change(ispin) # change of total magnetization after flip
        spin_idx = self.multi_idx(ispin) # find the spin to flip
        self._spins[spin_idx] *= -1 # flip the spin
        return dmag
    # end def flip_spin

    # end modifier methods
    # ---------------------------

    # ---------------------------
    # begin I/O methods

    def append_frame(self,filename):
        fhandle = open(filename,'a')
        fhandle.write("%d\n"%self.size())
        for irow in range(self.nside()):
            row_text = " ".join(["%2d"%spin for spin in self._spins[irow]])
            fhandle.write(row_text+"\n")
        # end for irow
        fhandle.close()
    # end def append_frame

    def load_frame(self,filename,ref_frame=-1,max_nframe=10**6):
        fhandle = open(filename,'r+')
        from mmap import mmap
        mm = mmap(fhandle.fileno(),0)

        # first line should specify the number of spins
        first_line  = mm.readline()
        nspin = int(first_line)
        mm.seek(0) # rewind file
        if nspin != self.size():
            raise InputError("number of spins in the trajectory file (%d) does not match the number of spins on this lattice %d"%(nspin,self.size()))
        elif int( (nspin)**(1./self.ndim()) ) != self.nside():
            raise InputError("number of spins in the trajectory file (%d) does not match lattice shape"%nspin)
        # end if

        # find starting positions of each frame in trajectory file
        frame_starts = []
        for iframe in range(max_nframe):
            # stop if end of file is reached
            if mm.tell() >= mm.size():
                break
            # end if

            # locate starting line (should be "%d\n" % nspin)
            idx = mm.find(first_line)
            if idx == -1:
                break
            # end if

            frame_starts.append(idx)
            mm.seek(idx)
            myline = mm.readline()

        # end for iframe

        # go to desired frame
        mm.seek(frame_starts[ref_frame])
        mm.readline() # skip first line with nspin

        # read spin states
        for irow in range(self.nside()):
            self._spins[irow] = np.array(mm.readline().split(),dtype=int)
        # end for

    # end def load_frame

    def visualize(self,ax):
        """ draw the lattice on a matplotlib axes object ax """
        if self._ndim != 2:
            raise NotImplementedError("visualization only implemented for 2D lattice")
        # end if

        # draw spins
        ax.pcolormesh(self.spins().T,vmin=-self._spin_mag,vmax=self._spin_mag)

        # set ticks to spin centers
        ax.set_xticks(np.arange(self.nside())+0.5)
        ax.set_yticks(np.arange(self.nside())+0.5)
        # rename ticks
        ax.set_xticklabels(np.arange(self.nside()))
        ax.set_yticklabels(np.arange(self.nside()))
    # end def

    # end I/O methods
    # ---------------------------

    def neighbours(self,ispin):
        """ return a list of indices pointing to the neighbours of spin ispin """
        spin_idx    = self.multi_idx(ispin)
        neighb_list = np.zeros(self._nb_per_dim*self.ndim(),dtype=int)
        #[up, down, left, right] as [0,1,2,3]
        length = self.nside()
        neighb_list[0] = self.linear_idx((spin_idx[0],(spin_idx[1]+1)%length))
        neighb_list[1] = self.linear_idx((spin_idx[0],(spin_idx[1]-1)%length))
        neighb_list[2] = self.linear_idx(((spin_idx[0]-1)%length,spin_idx[1]))
        neighb_list[3] = self.linear_idx(((spin_idx[0]+1)%length,spin_idx[1]))
        # !!!! IMPLEMENT THIS FUNCTION. done !!
        return neighb_list
    # end def neighbours

    def magnetization(self):
        """ total magnetization of the lattice """
        tot_mag = 0.0
        mag     = self.spin_mag()
        # !!!! IMPLEMENT THIS FUNCTION. done !!
        for i in range(self.size()):
            tot_mag += self.spin(i)
        tot_mag *= mag
        return tot_mag
    # end def

    def mag_change(self,ispin):
        """ change in magnetization if ispin is flipped """
        dmag     = 0.0
        # !!!! IMPLEMENT THIS FUNCTION. done!!
        dmag     = -2.0*self.spin(ispin)*self.spin_mag()
        return dmag
    # end def mag_change

# end class CubicIsingLattice

if __name__ == '__main__':

    # test flip_spin()
    lat = CubicIsingLattice(4)
    assert np.isclose(lat.spins()[1,1],1)
    lat.flip_spin(5) # row major -> second row, second column
    assert np.isclose(lat.spins()[1,1],-1)


    # for each spin, visualize its neighbours
    lat = CubicIsingLattice(4)
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(4,4)
    for ispin in range(lat.size()):

        # flip'em
        neighbours = lat.nb_list()[ispin]
        for nb in neighbours:
            lat.flip_spin(nb)
        # end for

        # plot'em
        spin_idx = lat.multi_idx(ispin)
        lat.visualize(ax[spin_idx])

        # flip'em back
        for nb in neighbours:
            lat.flip_spin(nb)
        # end for

    # end for ispin
    plt.show()

    # test magnetization()
    lat = CubicIsingLattice(4,spin_mag=1.0)
    assert np.isclose(16,lat.magnetization()), 'magnetization() incorrect'

    lat = CubicIsingLattice(5,spin_mag=0.5)
    assert np.isclose(12.5,lat.magnetization()), 'magnetization() incorrect'

    # test mag_change(ispin)
    lat = CubicIsingLattice(4,spin_mag=0.5)
    assert np.isclose(-1.0,lat.mag_change(5)), 'mag_change() incorrect'

# end if
