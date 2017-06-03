from __future__ import print_function
import random,math,pylab,numpy
L = 100
Nstep = 100000
beta = 0.3
neighbors = [(0,-1),(0,1),(-1,0),(1,0)]
#initialisation
s = [[ 1 for i in range(L)] for j in range(L)]
energy = -2. * L**2
toten = 0.
#evolution
for step in range(Nstep):
    kx = random.randrange(L)
    ky = random.randrange(L)
    h = sum( s[(kx+dx)%L][(ky+dy)%L] for (dx,dy) in neighbors)
    olds = s[kx][ky]
    Upsilon = random.random()
    s[kx][ky] = -1
    if Upsilon < 1./(1+math.exp(-2*beta*h)): s[kx][ky] = 1
    if olds != s[kx][ky]: energy-=2*h*s[kx][ky]
    toten += energy
print(toten / Nstep / L**2)
print(sum(sum(s[i][j] for i in range(L)) for j in range(L) )/L**2.)
pylab.matshow(numpy.array(s))
pylab.show()
