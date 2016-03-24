import numpy as np
from cmath import phase
from scipy.optimize import brentq as findroot

h=1.7
L=20


def phi(k):
    global h
    return phase(complex(-np.sin(k), np.cos(k)-h))

def cond(k,i):
    global L
    return ((-2*i+1)*np.pi + 2*phi(k)) / (2*(L+1)) - k


for n in xrange(2*L):
    print findroot(cond,-np.pi, np.pi, args=n)
    
