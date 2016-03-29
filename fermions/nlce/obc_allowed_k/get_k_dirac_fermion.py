import numpy as np
from cmath import phase
from scipy.optimize import root
import math

L=20

def theta_j(h, abs_h):
    return np.acos( h[2] / abs_h)

def phi_j(h, abs_h):
    return np.asin( h[0] / (abs_h*np.sin(theta(h, abs_h)) ) )

def phi(kx, ky):
    return phase( complex( np.sin(kx), np.sin(ky) ) ) 

def vec_func1(k, mn):
    global Lx, Ly
    kx, ky = k
    m, n = mn

    cond_x = kx * (Lx + 1) - m * np.pi - phi(ky, kx)
    cond_y = ky * (Ly + 1) - n * np.pi - phi(kx, ky)

    return (cond_x, cond_y)



Lx = 10
Ly = 10

result_list=[]

for n in xrange(1,Ly+1):
    for m in xrange(1,Lx+1):
        Kx, Ky = root(vec_func1,(np.pi/2, np.pi/2), args=([m,n]) ).x
        result_list.append([Kx, Ky])

for x, y in result_list:
    print x,y
