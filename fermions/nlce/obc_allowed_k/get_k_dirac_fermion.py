import numpy as np
from cmath import phase
from scipy.optimize import brentq as findroot

L=20

def theta(h, abs_h):
    return np.acos( h[2] / abs_h)

def phi(h, abs_h):
    return np.asin( h[0] / (abs_h*np.sin(theta(h, abs_h)) ) )


def cond(k,i):



h=np.array([np.sin(kx), np.sin(ky), 2-np.cos(kx)-np.cos(ky)])
abs_h=np.sqrt( h**2 )

for n in xrange(2*L):
    print findroot(cond,-np.pi, np.pi, args=n)
    
