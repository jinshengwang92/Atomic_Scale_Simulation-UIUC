#!/usr/bin/env python

from __future__ import print_function

import numpy as np

global mymean
global mystd
global myerr
global mycorr

def mean(trace):
    """ calculate the mean of a trace of scalar data
    results should be identical to np.mean(trace)
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the mean of this trace of scalars
    """

    total = 0.0        # store sum of all data points
    num   = len(trace) # count total number of data points

    for i in range(num):
        total += trace[i]

    global mymean
    mymean = total/float(num)
    return total/float(num)

# end def mean

def std(trace):
    """ calculate the standard deviation of a trace of scalar data
    results should be identical to np.std(trace)
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the standard deviation of this trace of scalars
    """

    stddev = 0.0
    num    = len(trace)
    sum_of_square = 0.0
    for i in range(num):
        sum_of_square += (trace[i]-mymean)**2

    stddev = np.sqrt(sum_of_square/float(num-1))
    # calculate stadard deviation

    global mystd
    mystd = stddev
    return stddev

# end def std

def corr(trace):
    """ calculate the autocorrelation of a trace of scalar data
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the autocorrelation of this trace of scalars
    """

    def auto_corr(k_value):
        num = len(trace) - k_value
        variance = mystd**2
        sum = 0
        ave =  mymean
        for i in range(num):
            sum += (trace[i]-ave)*(trace[i+k_value]-ave)

        return float(sum)/num/variance

    k_trial = 1
    for i in range(1,len(trace)):
        if auto_corr(i) > 0:
            k_trial = i
        else:
            break

    correlation = 1.0
    for k in range(1,k_trial+1):
        correlation += 2*auto_corr(k)

    # calculate auto correlation
    global mycorr
    mycorr = correlation
    return correlation

# end def corr

def error(trace):
    """ calculate the standard error of a trace of scalar data
    for uncorrelated data, this should match np.std(trace)/np.sqrt(len(trace))
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return the standard error of this trace of scalars
    """
    num = len(trace)
    num_eff = num/mycorr
    stderr = mystd/np.sqrt(num_eff)
    # calculate standard error
    global myerr
    myerr = stderr
    return stderr

# end def error

def stats(trace):
    """ return statistics of a trace of scalar data
    pre:  trace should be a 1D iterable array of floating point numbers
    post: return (mean,stddev,auto_corr,error)

    basically a composition of functions implemented above
    """
    global mymean
    global mystd
    global mycorr
    global myerr
    mymean = mean(trace)
    mystd  = std(trace)
    mycorr = corr(trace)
    myerr  = error(trace)
    return (mymean,mystd,mycorr,myerr)
# end def stats

if __name__ == '__main__':
    """ code protected by __main__ namespace will not be executed during import """
    import argparse
    global mymean
    global mystd
    global mycorr
    global myerr

    # parse command line input for trace file name
    parser = argparse.ArgumentParser(description='analyze a trace')
    parser.add_argument('filename', type=str, help='filename containing a scalar trace')
    parser.add_argument('-p','--plot', action='store_true', help='plot data')
    parser.add_argument('-i','--initial_index', type=int, default=0, help='initial index of data to include')
    parser.add_argument('-f','--final_index', type=int, default=-1, help='final index of data to include, must be larger than initial_index')
    args = parser.parse_args()
    trace_file_name = args.filename

    # read trace file
    trace = np.loadtxt( trace_file_name )
    print('*'*80)
    print('np_mean  ='+str(np.mean(trace)))
    print('np_std   ='+str(np.std(trace)))
    print('np_stderr='+str(np.std(trace)/np.sqrt(len(trace))))
    # determine final cutoff
    final_index = -1; # default
    if args.final_index == -1:
        final_index = len(trace)
    else:
        final_index = args.final_index
    # end if

    # guard against misuse
    if final_index > len(trace):
        raise RuntimeError("final_index exceeds length of trace")
    # end if

    # cut out interested portion
    trace = trace[args.initial_index:final_index]

    # calculate statistics
    mymean,mystd,mycorr,myerr = stats( trace )

    # formatted output of calculated statistics
    header = "%25s       mean       stddev       corr       err" % "filename"
    fmt    = "{filename:s}  {mean:1.8f}  {stddev:1.8f}   {corr:1.8f}  {err:1.8f}"
    output = fmt.format(
            filename = trace_file_name
          , mean     = mymean
          , stddev   = mystd
          , corr     = mycorr
          , err      = myerr )

    print( header )
    print( output )
    print('*'*80)
    if (args.plot):

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            1,2, gridspec_kw = {'width_ratios':[3, 1]})
        ax[0].set_xlabel("index", fontsize=14)
        ax[0].set_ylabel("data" , fontsize=14)
        ax[1].set_xlabel("freq.", fontsize=14)
        ax[1].get_yaxis().tick_right()

        # plot trace
        ax[0].plot(trace,c='black')

        # plot histogram
        wgt,bins,patches = ax[1].hist(trace, bins=30, normed=True
            , fc='gray', alpha=0.5, orientation='horizontal')
        # moving averge to obtain bin centers
        bins = np.array( [(bins[i-1]+bins[i])/2. for i in range(1,len(bins))] )
        def _gauss1d(x,mu,sig):
            norm  = 1./np.sqrt(2.*sig*sig*np.pi)
            gauss = np.exp(-(x-mu)*(x-mu)/(2*sig*sig))
            return norm*gauss
        # end def
        ax[1].plot(_gauss1d(bins,mymean,mystd),bins,lw=2,c="black")
        ax[1].set_xticks([0,0.5,1])

        # overlay statistics
        for myax in ax:
            myax.axhline( mymean, c='b', lw=2, label="mean = %1.4f" % mymean )
            myax.axhline( mymean+mystd, ls="--", c="gray", lw=2, label="std = %1.2f" % mystd )
            myax.axhline( mymean-mystd, ls="--", c="gray", lw=2 )
        # end for myax
        ax[0].legend(loc='best')

        plt.show()

    # end if plot

# end __main__
