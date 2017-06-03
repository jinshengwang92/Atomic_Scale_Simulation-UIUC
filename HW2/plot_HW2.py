# plot the mean and stddev of TE for different time steps dt

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt


'''
# first time without stddev
x = (0.01,0.04,0.08,0.12,0.16,0.17,0.18,0.19,0.20,0.21)
mean = (-257.6773056,-257.7390748,-257.6547991,-257.5120141,-257.1081079,-256.4068432,-256.4238266,-256.483974,-253.1829523,-245.2093645)
stddev = (0.003109922,0.02768503,0.10572314,0.23219395,0.45823917,0.76109949,0.70874196,1.43078724,2.98065495,9.42455094)

fig = plt.figure()
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(x,mean,'bs',label = 'Mean of TE')
ax2.plot(x,stddev,'r^',label = 'Stddev of TE')
ax1.set_xlabel('Time step dt',fontsize=16)
ax1.set_ylabel('Mean',fontsize=16)
ax2.set_ylabel('Stddev',fontsize=16)
ax1.legend(loc='upper left')
ax2.legend(loc='upper center')
plt.title('TE data for different time step dt')
ax1.grid()
plt.grid(True)
plt.show()
'''

#plot for mean
fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
x = (0.01,0.04,0.08,0.12,0.16,0.17,0.18,0.19,0.20,0.21)
ping = (-257.6808826,-257.7307407,-257.6803141,-257.535705,-256.9985424,-256.7345144,-256.5175695,-256.1983878,-255.5644379,-250.5822685)
error1 = (0.00306951,0.02924268,0.10811419,0.23688659,0.5211398,0.63099862,0.75957636,1.28984697,1.89517032,4.03453415)
ax0.grid(True)
ax0.errorbar(x, ping, yerr = error1,fmt='-o')
ax0.set_title('Mean of TE variation w.r.t. different time step dt')
ax0.set_xlabel('dt')
ax0.set_ylabel('Mean TE value')

stddevv1 = (0.00306951,0.02924268,0.10811419,0.23688659,0.5211398,0.63099862,0.75957636,1.28984697,1.89517032,4.03453415)
error2 = (0.00003212,0.00158921,0.00482222,0.00620743,0.08539932,0.09375007,0.15500545,0.18090759,0.79801777,1.19933565)
ax1.errorbar(x,stddevv1,yerr = error2,fmt='-o')
ax1.set_title('Stddev of TE variation w.r.t. different time step dt')
ax1.set_xlabel('dt')
ax1.set_ylabel('Stddev TE value')
ax1.grid(True)
plt.show()

# end of plot
