# -*- encoding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns


'''
========== HW4 of atomic scale simulation ============
========== Jinsheng Wang, NetID:jwang278  ============
'''

# def of LCG for next pseudo random number generator
def LCG(m,a,c,xn):
    next_xn = (a*xn + c)%m   #take the mudulus
    return next_xn
#end def LCG

# def of generation of all the Random numbers in integers (0,m)
def all_RN_in(seed,num,m,a,c):
    RNs    = np.zeros(num)
    RNs[0] = LCG(m,a,c,seed)
    for i in range(1,num):
        RNs[i] = LCG(m,a,c,RNs[i-1]) #generate next RN
    #end for
    return RNs
#end def of all_RN_in

# def of generation of all the RNs in range [0,1)
def all_RN_01(seed,num,m,a,c):
    RNs    = np.zeros(num)
    RNs[0] = LCG(m,a,c,seed)
    for i in range(1,num):
        RNs[i] = LCG(m,a,c,RNs[i-1]) #generate next RN
    #end for
    RNs = np.array([i*1.0/m for i in RNs])
    return RNs
#end def of all_RN

# def of generation of all the RNs in range [-0.5,0.5)
def all_RN_55(seed,num,m,a,c):
    RNs    = np.zeros(num)
    RNs[0] = LCG(m,a,c,seed)
    for i in range(1,num):
        RNs[i] = LCG(m,a,c,RNs[i-1]) #generate next RN
    #end for
    RNs = np.array([i*1.0/m-0.5 for i in RNs])
    return RNs
#end def of all_RN

#def of generating normal distribution with certain mean and stddev with
#Marsaglia polar method
def NRNG(m,a,c,seed,num,mu,sigma):
    #seed is the starter for 0,1 distribution
    #num is the total number of normal Random numbers
    #mu is the mean
    #sigma is the stddev of distribution
    ans = np.zeros(num)     #array used to store all the normal RNs
    RN_0  = LCG(m,a,c,seed) #first value for LCG, global var for the if loop
    RN_1  = 0.0     # used to store (0,m) intergers, global for if loop
    RN_2  = 0.0     # used to store (0,m) intergers, global for if loop
    count = 0       # add by 2 every time to numerate ans
    while count < num:
        RN_1   = LCG(m,a,c,RN_0)
        x      = (LCG(m,a,c,RN_1)-m/2.0)*2.0/m  # x in range(-1,1)
        RN_2   = LCG(m,a,c,RN_1)
        y      = (LCG(m,a,c,RN_2)-m/2.0)*2.0/m  # y in range(-1,1)
        RN_0   = RN_2
        s = x*x + y*y
        if s < 1:
            ans[count]   = x*np.sqrt(-2.0*np.log(s)/s)*sigma + mu
            ans[count+1] = y*np.sqrt(-2.0*np.log(s)/s)*sigma + mu
            count += 2
        #end if
    #end if to finish generating all the RNs
    return ans
#end def of NRNG

#def the function to plot normal distribution and compare with anal. result
def plot_scatter(L,num_bin,rns):
    #(-L,L) is the range for histogram
    #num_bin is the num of bin each side of y peak, total 2*num_bin
    #rns is the RNs we want to test
    N   = len(rns)
    bin1 = L*1.0/num_bin
    y_t   = np.zeros(2*num_bin+1)  # for temperarilly use
    x_t = np.linspace(-num_bin,num_bin,2*num_bin+1) # for temperarilly use
    x   = np.array([i*bin1 for i in x_t])  #total 2*num_bin+1 steps
    for val in rns:
        index = int(round(1.0*val/bin1))
        y_t[index+num_bin] += 1
    #end for
    y   = np.array([i*(2*num_bin+1)/2.0/L/N for i in y_t])

    #plot for x,y and standard formula
    y2  = np.array([1.0/np.sqrt(2*np.pi)*np.exp(-i*i/2.0) for i in x])
    plt.figure('RNG_vs_equation_scatter')
    plt.plot(x,y2,label='standard Normal distribution')
    plt.scatter(x,y,label='scatter point of my Norm RNs')
    plt.title('my RNG vs analytical equation')
    plt.xlabel('random numbers')
    plt.ylabel('frquency')
    plt.legend()
    #plt.savefig('RNG_vs_equation_scatter.png')
    plt.show()
#end def plot_scatter

#def the function to plot normal distribution and compare with anal. result
def plot_histo(L,num_bin,rns):
    #(-L,L) is the range for histogram
    #num_bin is the num of bin each side of y peak, total 2*num_bin
    #rns is the RNs we want to test
    #N   = len(rns)
    bin1 = L*1.0/num_bin
    x_t = np.linspace(-num_bin,num_bin,2*num_bin+1) # for temperarilly use
    x   = np.array([i*bin1 for i in x_t])  #total 2*num_bin+1 steps

    #plot for x,y and standard formula
    y2  = np.array([1.0/np.sqrt(2*np.pi)*np.exp(-i*i/2.0) for i in x])
    plt.figure('RNG_vs_equation_histo')
    plt.plot(x,y2,label='standard Normal distribution')
    plt.hist(rns,bins=2*num_bin+1,normed=True,range=(-L,L),
            label='histogram of my Norm RNs')
    plt.title('my RNG vs analytical equation')
    plt.xlabel('random numbers')
    plt.ylabel('frquency')
    plt.legend()
    #plt.savefig('RNG_vs_equation_histo.png')
    plt.show()
#end def plot_histo

#def of CheckRandomNumbers1D
def CheckRandomNumbers1D(numberlist,pointsPerDim):
    #take a list of random numbers
    #return the chi_squared
    hist    = np.zeros(pointsPerDim)
    N       = len(numberlist)
    n_i     = 1.0*N/pointsPerDim    #expected number of counts in bin i
    # test and make sure that n_i is greater than 5
    assert(n_i > 5), 'n_i is not greater than 5: %s' %str(n_i)
    binning = 1.0/pointsPerDim
    for val in numberlist:
        index        = int(val/binning)
        hist[index] += 1
    #end for
    chi_squared = 0.0
    for val in hist:
        chi_squared += 1.0*(val - n_i)*(val - n_i)/n_i
    #end for
    return chi_squared
#end def of CheckRandomNumbers1D

#def of CheckRandomNumbers2D
def CheckRandomNumbers2D(numberlist,pointsPerDim):
    #take a list of random numbers
    #return the chi_squared
    hist = np.zeros((pointsPerDim,pointsPerDim))
    N       = len(numberlist)
    n_i     = 1.0*N/pointsPerDim/pointsPerDim/2.0  #expected num of counts in bin i
    # test and make sure that n_i is greater than 5
    assert(n_i > 5), 'n_i is not greater than 5: %s' %str(n_i)
    binning = 1.0/pointsPerDim
    ii      = 0
    while  (ii+1) < N:
        index1 = int(numberlist[ii]/binning)
        index2 = int(numberlist[ii+1]/binning)
        hist[index1][index2] += 1
        ii    += 2
    #end while
    chi_squared = 0.0
    for i in range(pointsPerDim):
        for j in range(pointsPerDim):
            chi_squared += (hist[i][j] - n_i)**2/n_i

    #end for
    return chi_squared
#end def of CheckRandomNumbers2D

#def of CheckRandomNumbers3D
def CheckRandomNumbers3D(numberlist,pointsPerDim):
    #take a list of random numbers
    #return the chi_squared
    hist = np.zeros((pointsPerDim,pointsPerDim,pointsPerDim))
    N       = len(numberlist)
    n_i     = 1.0*N/pointsPerDim/pointsPerDim/pointsPerDim/3.0
    #expected num of counts in bin i
    #test and make sure that n_i is greater than 5
    assert(n_i > 5), 'n_i is not greater than 5: %s' %str(n_i)
    binning = 1.0/pointsPerDim
    ii      = 0
    while  (ii+2) < N:
        index1 = int(numberlist[ii]/binning)
        index2 = int(numberlist[ii+1]/binning)
        index3 = int(numberlist[ii+2]/binning)
        hist[index1][index2][index3] += 1
        ii    += 3
    #end while
    chi_squared = 0.0
    for i in range(pointsPerDim):
        for j in range(pointsPerDim):
            for k in range(pointsPerDim):
                chi_squared += (hist[i][j][k] - n_i)**2/n_i
    #end for
    return chi_squared
#end def of CheckRandomNumbers3D

#def function to return 1D, 2D, and 3D chi_squared values for fiven numberlist
def cal_1D_2D_3D_chisqu(numberlist, ppd):   #ppd means points per dimension
    chi1 = CheckRandomNumbers1D(numberlist,ppd)
    chi2 = CheckRandomNumbers2D(numberlist,ppd)
    chi3 = CheckRandomNumbers3D(numberlist,ppd)
    print('1D chi_square = %f'%chi1)
    print('2D chi_square = %f'%chi2)
    print('3D chi_square = %f'%chi3)
#end def cal_1D_2D_3D_chisqu

#def of function to plot the 2D bin numbers,
#run for first 1000 steps of supplied datasets
def Plot2d(numberList,pointsPerDim):
    # makes a plot of the pairs of random numbers in 2d
    x = [numberList[i] for i in range(0,len(numberList),2)]
    y = [numberList[i] for i in range(1,len(numberList),2)]
    plt.hist2d(x,y,bins=pointsPerDim,label='1000 (0,1) RNS in 2D bins')
    plt.colorbar()
    plt.show()
    return None
#end def of Plot2d

#def of integral by numerical quadrature
def NumIntegral(L,N):
    #(-L,L) is the area for calcualtion
    #N the number of grid points
    L = L*1.0
    N = int(N)
    assert(N%2 == 0), 'N should be even value here, your N = %d'%N
    area  = 0.0     # use to store the final calculus
    del_x = 2.0*L/N     #grid resolution
    a     = range(0,N)
    shift = (N-1)*L/N
    xbin  = [val*del_x-shift for val in a]
    for x in xbin:
        area += np.exp(-x*x/2.0)/(1+x*x)*del_x
    #end for
    return area
#end def NumIntegral

#def MC integral
def CalcIntegral(alpha,N):
    #calculates entire integral
    #returns the integral and the variance!
    #NRNG(m,a,c,seed,num,mu,sigma)
    #N is the total number of RNs
    RNGs      = NRNG(2**32,69069,1,0,N,0.0,np.sqrt(alpha))
    mean1     = 0.0 #to store mean value
    variance1 = 0.0 #to store variance
    g_x       = np.zeros(N) # to store the estimator
    for i in range(N):
        x      = RNGs[i]
        g_x[i] = np.sqrt(2.0*np.pi*alpha)/(1.0+x*x)*np.exp(-x*x/2.0*(1.0-1.0/alpha))
    #end for
    mean1      = np.mean(g_x)
    variance1  = np.var(g_x)
    return (mean1,variance1)
#end def CalcIntegral

#def of plot of alpha vs variance
def plot_alp_vs_var(x,var1,var2):
    var3 = np.zeros(len(x))
    for i in range(len(x)):
        var3[i] = var2[i]/var1[i]
    #end for
    #subplot1
    fig = plt.figure(figsize=(6,6))
    ax1 = fig.add_subplot(311)
    ax1.plot(x,var1)
    ax1.scatter(x,var1)
    ax1.set_title(r'variance vs alpha using 10^6 RNs')
    ax1.set_ylabel('variance')
    ax1.grid()
    #subplot2
    ax2 = fig.add_subplot(312)
    ax2.plot(x,var2)
    ax2.scatter(x,var2)
    ax2.set_title(r'variance vs alpha using 10^7 RNs')
    ax2.set_ylabel('variance')
    ax2.grid()
    #subplot3
    ax3 = fig.add_subplot(313)
    ax3.plot(x,var3)
    ax3.scatter(x,var3)
    ax3.plot([min(x),max(x)],[1,1])
    ax3.set_title(r'variance ratio vs alpha')
    ax3.set_xlabel('alpha')
    ax3.set_ylabel('ratio of var2/var1')
    ax3.grid()
    fig.subplots_adjust()
    plt.show()
#end def plot_alp_vs_var

#def g_x_alpha()
def g_x_alpha(alpha, N):
    #calculates entire integral
    #returns the integral and the variance!
    #NRNG(m,a,c,seed,num,mu,sigma)
    #N is the total number of RNs
    RNGs      = NRNG(2**32,69069,1,0,N,0.0,np.sqrt(alpha))
    g_x       = np.zeros(N) # to store the estimator
    for i in range(N):
        x      = RNGs[i]
        g_x[i] = np.sqrt(2.0*np.pi*alpha)/(1.0+x*x)*np.exp(-x*x/2.0*(1.0-1.0/alpha))
    #end for
    return(g_x)
#end def

#start of the main funcion
if __name__ == '__main__':


    #===   uniform RNG and Gaussian Random Number Generator   ======
    #print for LCG(16,3,1,2) interger RNs
    RN_in_all_1 = all_RN_in(2,20,16,3,1)
    print(r'LCG(16,3,1,2) for 20 steps')
    print(RN_in_all_1)

    #print for LCG(2^32,69069,1,0) as intergers
    RN_in_all_2 = all_RN_in(0,10,2**32,69069,1)
    print(r'LCG(2**32,69069,1,0) intergers for 10 steps')
    print(RN_in_all_2)

    #print for LCG(2^32,69069,1,0) in [0,1)
    RN_all_01_1 = all_RN_01(0,10,2**32,69069,1)
    print(r'LCG(2**32,69069,1,0) in (0,1)for 10 steps')
    print(RN_all_01_1)

    #print for LCG(2^32,69069,1,0) in (-0.5,0.5)
    RN_all_55_1 = all_RN_55(0,10,2**32,69069,1)
    print(r'LCG(2**32,69069,1,0) in (-0.5,0.5)for 10 steps')
    print(RN_all_55_1)

    #print for 100 Normal distribution RNs
    RNGs = NRNG(2**32,69069,1,0,1000000,0.0,1.0)
    plt.hist(RNGs,bins=35)

    #plot the figure for my RNG vs anaytical equations
    #plot_histo(L,num_bin,rns)
    plot_histo(5.0,100,RNGs)
    #plot_scatter(L,num_bin,rns)
    plot_scatter(5.0,100,RNGs)
    #================================================================



    #===========    Testing Random Number Generators   ==============
    #test chi_squared for my (0,1) RNGs
    RN_all_01_1 = all_RN_01(0,1000000,2**32,69069,1)
    cal_1D_2D_3D_chisqu(RN_all_01_1, 16)
    #end test

    #test chi_squared for random.random() (0,1) RNS
    ran_ran_RNs = np.array([np.random.random() for i in range(1000000)])
    cal_1D_2D_3D_chisqu(ran_ran_RNs, 16)
    #end test

    #test chi_squared for given files (0,1) RNS
    #use numpy.loadtxt('Generator1.dat') to call 3 different files
    given_RNs1 = np.loadtxt('Generator1.dat')
    cal_1D_2D_3D_chisqu(given_RNs1, 16)
    given_RNs2 = np.loadtxt('Generator2.dat')
    cal_1D_2D_3D_chisqu(given_RNs2, 16)
    given_RNs3 = np.loadtxt('Generator3.dat')
    cal_1D_2D_3D_chisqu(given_RNs3, 16)
    #end test

    #plot using Plot2d(numberList,pointsPerDim)
    data1_1000 = np.loadtxt('Generator1.dat')[:1000]
    Plot2d(data1_1000,10)
    data2_1000 = np.loadtxt('Generator2.dat')[:1000]
    Plot2d(data2_1000,10)
    data3_1000 = np.loadtxt('Generator3.dat')[:1000]
    Plot2d(data3_1000,10)
    #end of plot plot2d
    #=================================================================



    #=====================  importance sampling  ====== ==============

    #numerical calculus vs L
    L_val = [1,2,3,4,5,6,7]
    area  = np.zeros(len(L_val))
    for i in range(len(L_val)):
        area[i] = NumIntegral(L_val[i],10000)
        print('calculus = %.6f when L = %.1f'%(area[i],L_val[i]))
    #end for
    plt.plot(L_val,area)
    plt.scatter(L_val,area)
    plt.plot([0, max(L_val)+0.5], [1.643545, 1.643545], '--r')
    plt.title('Numerical calculus vs L')
    plt.xlabel('L')
    plt.xlim(0,max(L_val)+0.5)
    plt.ylabel('calculus values')
    plt.grid()
    #plt.show()


    #MC calculus with CalcIntegral(alpha,N)
    alp1 = [0.1,0.2,0.3,0.4,0.5,0.6,0.9,1.3,1.7]
    N_all= [10**6,4*10**6]
    for AL in alp1:
        for N in N_all:
            a = CalcIntegral(AL,N)
            print('alpha=%f，N=%d: integral=%f, variance=%f' %(AL,N,a[0],a[1]))
        #end for
    #end for


    #plot alpha vs variance
    #input variables
    x    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.3, 1.7] #x axis
    y1   = [18.02,2.252,0.3329,0.05403,0.05691,0.3324,0.7602,1.1607]  #y1 axis
    y2   = [56.44,4.375,0.4671,0.06334,0.05750,0.3324,0.7602,1.1603]  #y2 axis
    plot_alp_vs_var(x,y1,y2)
    #en of plot alpha vs variance

    #plot g_x_alpha(alpha, N) for different alpha
    Numb = 10000  # num of RNS
    alp  = [0.2,0.6]  # alpha
    g1 = g_x_alpha(alp[0],Numb)
    g2 = g_x_alpha(alp[1],Numb)
    fig = plt.figure('g_x_alpha comparison')
    plt.plot(range(Numb),g1,label='alpha = %.2f'%alp[0])
    plt.plot(range(Numb),g2,label='alpha = %.2f'%alp[1])
    plt.xlabel('Num of RNs')
    plt.ylabel('g_x_alpha')
    plt.legend()
    plt.title('g_x_alpha comparison')
    plt.show()
    #end of plot g_x_alpha(alpha, N)
    #=================================================================

#end def main function
