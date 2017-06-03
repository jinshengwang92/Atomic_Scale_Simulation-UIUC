import matplotlib.pyplot as plt


x = [0.8, 1.6, 1.8, 2.0,2.2,2.3,2.4,3.2,4.0]
y = [19.5,15.0,14.8,10.5,9.5,9.5,10,5.1,2.6]

plt.plot(x,y,label='first peak of S(k)')
plt.scatter(x,y)
plt.legend()
plt.xlabel('temperature')
plt.ylabel('peak strength')
plt.title('First peak intensity of S(k)')
plt.grid()
plt.show()
