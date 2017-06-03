# -*- encoding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

x    = [0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.3, 1.7]
var1 = [18.02,2.252,0.3329,0.05403,0.05691,0.3324,0.7602,1.1607]
var2 = [56.44,4.375,0.4671,0.06334,0.05750,0.3324,0.7602,1.1603]
var3 = np.zeros(len(x))
for i in range(len(x)):
    var3[i] = var2[i]/var1[i]
#end for
fig = plt.figure(figsize=(6,6))
ax1 = fig.add_subplot(311)
ax1.plot(x,var1)
ax1.scatter(x,var1)
ax1.set_title(r'variance vs alpha using 10^6 RNs')
ax1.set_ylabel('variance')
ax1.grid()

ax2 = fig.add_subplot(312)
ax2.plot(x,var2)
ax2.scatter(x,var2)
ax2.set_title(r'variance vs alpha using 10^7 RNs')
ax2.set_ylabel('variance')
ax2.grid()

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
