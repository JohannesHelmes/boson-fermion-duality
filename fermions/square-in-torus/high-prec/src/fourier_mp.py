####################################################################################
# inverse Fourier transformation in 2D using standard floating point precision
#
# (c) Johannes Helmes 2015
#####################################################################################
from numpy.fft import *
import numpy as np

def my_ift2_radix2_divide(qx):
    M,N=qx.shape
    assert M==N

    if M%2==1:
        return my_ift2_64(qx)
    ft=np.zeros((M,N),dtype='complex')
    eveneven=my_ift2_radix2_divide(qx[::2,::2])
    evenodd=my_ift2_radix2_divide(qx[::2,1::2])
    oddeven=my_ift2_radix2_divide(qx[1::2,::2])
    oddodd=my_ift2_radix2_divide(qx[1::2,1::2])

    factors_list=[np.exp(2*np.pi*1j*m/M) for m in range(M/2)]  #arange here!!! /N
    W=np.array([factors_list])

    ft[0:M/2,0:M/2]= eveneven + evenodd * W + W.T * oddeven + W.T * oddodd * W
    ft[0:M/2,M/2:]= eveneven - evenodd * W + W.T * oddeven - W.T * oddodd * W
    ft[M/2:,0:M/2]= eveneven + evenodd * W - W.T * oddeven - W.T * oddodd * W
    ft[M/2:,M/2:]= eveneven - evenodd * W - W.T * oddeven + W.T * oddodd * W
    return ft/4 #Division by factor of 4 = 2 * 2 because of reverse Fourier transform, for each recursion step


def my_ift2_64(qx):
    ft=np.zeros(qx.shape,dtype='complex')
    M,N=qx.shape

    phase=2*np.pi*1j
    for k in np.arange(M):
        for l in np.arange(N):
            for m in np.arange(M):
                for n in np.arange(N):
                    weight=float(m*k)/M + float(n*l)/N
                    ft[k,l]+=qx[m,n]*np.exp(phase*weight)

    ft/=(M*N)
    return ft

if __name__=="__main__":
    N=16
    random_data=np.random.random((N,N))
    #built_in_ifft2=ifft2(random_data)
    print "doing brute force"
    my_ift2=my_ift2_64(random_data)
    print "My implementation", my_ift2
    my_ift2_radix2_divide=my_ift2_radix2_divide(random_data)
    #print "Built in", built_in_ifft2
    print "My simple FFT", my_ift2_radix2_divide
    print "Test"
