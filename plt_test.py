import matplotlib
matplotlib.use('tkagg')
 
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi,np.pi,100)
s = np.sin(x)


plt.plot(s)
plt.show()
