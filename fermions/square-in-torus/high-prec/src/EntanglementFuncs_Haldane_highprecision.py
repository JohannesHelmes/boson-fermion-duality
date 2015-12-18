#!/usr/bin/env python
# encoding: utf-8
"""
EntanglementFuncs_Haldane.py

Created by Anushya Chandran on 2015-06-30.

Wrapped to mpmath for high precision by Johannes Helmes on 2015-10-20.

"""


import os
import mpmath as mpm
import numpy as np
import time
mpm.mp.dps=33
eps=mpm.mpf('10')**(-mpm.mp.dps)
sigx = mpm.matrix([ [0,1], [1,0] ])
sigy = mpm.matrix([ [0, -1j], [ 1j,0] ])
sigz = mpm.matrix([ [1, 0], [0, -1]])

def one_chern_evec(Hamargs, kx, ky, chkspec= False):
  global sigx, sigy, sigz
  """H(k) = \vec{d}(k).\vec{\sigma}. This is the Hamiltonian in momentum space of a two band model. d(k) defines the entire model.
  Returns the energy and eigenstate of lower energy. 
  d_x = sin(k_x)
  d_y = sin(k_y)
  d_z = 2 + M - cos(k_x) - cos(k_y)
  where M is a parameter that decides the phases of the model. kx = momentum along x-direction. ky=momentum along y-direction.
  Phases:
  M>0 and M<-4: Trivial
  -2<M<0: Chern number of lowest band = -1
  -4<M<-2: Chern number of lowest band = 1
  
  chkspec: If true, then returns the entire energy spectrum at (kx,ky)
  
  Hamargs: Arguments to this Hamiltonian.
    Contains M defined below.
  """
  M = Hamargs[0]

  d = mpm.matrix([mpm.sin(kx) , mpm.sin(ky), 2 + M - mpm.cos(kx) - mpm.cos(ky)])
  
  
  darr = d[0]*sigx + d[1]*sigy + d[2]*sigz

  
  en, evecs = mpm.eighe(darr)

  evecs0=mpm.matrix([mpm.mpc(x) for x in evecs[:,0]])
  
  if chkspec:
    return en
  else:
    return (en[0], evecs0)# return only the lower energy and corresponding eigenstate

def square(center_coord, rad, Lx, Ly):
    """
    center_coord: tuple with (row, col) index of the center of the square
    rad: Length of the side of the square
    Lx: total number of rows
    Ly: total number of columns
    
    Returns a Lx X Ly array with 1s where the square is.
    """
    
    edgex = int(Lx/2.) - int(rad/2)
    edgey = int(Ly/2.) - int(rad/2)
    
    mask = np.zeros((Lx,Ly), dtype='bool')
    mask[edgex:edgex + int(rad), edgey:edgey + int(rad)] = True
    
    centerx = int(Lx/2.)
    centery = int(Ly/2.)
        
    tmpmask = np.roll(mask, int(np.floor(center_coord[1])) - centery, axis=1)
    return np.roll(tmpmask, int(np.floor(center_coord[0])) - centerx, axis=0)
    
    
'''
def cmatfull_analytic(M, bc, Lx, Ly):
    
    kylist = pi/Ly*arange(1,2*Ly+1,2)
    
    if bc=='pbc':
        kxlist = 2*pi/Lx*arange(Lx)
    elif bc=='apbc':
        kxlist = pi/Lx*arange(1,2*Lx+1,2)
    else:
        raise Exception("The flag bc should be 'pbc' or 'apbc'")
        
    (kygrid, kxgrid) = meshgrid( kylist, kxlist )   
    
    dx = sin(kxgrid)
    dy = sin(kygrid)
    dz = 2 + M - cos(kxgrid) - cos(kygrid)
    
    modd = sqrt(dx**2 + dy**2 + dz**2)
    
    cdagc = ifft2(- 0.5*dz/modd)
    cdagc[0,0] += 0.5
    ddagd = -cdagc
    ddagd[0,0] = 1.0 - cdagc[0,0]
    
    
    cdagd = -1*ifft2( (dx + 1j*dy)/(2*modd) )
    ddagc = -1*ifft2( (dx - 1j*dy)/(2*modd) )
    
    
    return [[cdagc, cdagd], [ddagc, ddagd]]
'''
    
def mat_conj(A):
    Aconj=mpm.zeros(A.rows,A.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            Aconj[i,j]=mpm.conj(A[i,j])

    return Aconj




def ift2_prec_radix2_divide(qx):
    global W
    M,N=qx.rows,qx.cols
    #assert M==N

    if M%2==1:
        return ift2_prec(qx)
    #ft=np.zeros((M,N),dtype='complex')
    ft=mpm.zeros(M,N)
    eveneven=ift2_prec_radix2_divide(qx[::2,::2])
    evenodd=ift2_prec_radix2_divide(qx[::2,1::2])
    oddeven=ift2_prec_radix2_divide(qx[1::2,::2])
    oddodd=ift2_prec_radix2_divide(qx[1::2,1::2])

    #factors_list=[np.exp(2*np.pi*1j*m/M) for m in range(M/2)]  #arange here!!! /N
    #W=np.array([factors_list])
    #W=mpm.matrix([mpm.exp(phase*m) for m in mpm.matrix(mpm.arange(M/2))/mpm.mpf(M)])

    weighted_evenodd=piecewise_mat_times_vec(evenodd,W[M,:M/2].T)
    weighted_oddeven=piecewise_vec_times_mat(W[M,:M/2] , oddeven)
    weighted_oddodd=piecewise_vec_times_mat(W[M,:M/2] , piecewise_mat_times_vec(oddodd , W[M,:M/2].T) )

    ft[0:M/2,0:M/2]= eveneven + weighted_evenodd + weighted_oddeven + weighted_oddodd
    ft[0:M/2,M/2:]= eveneven - weighted_evenodd + weighted_oddeven - weighted_oddodd
    ft[M/2:,0:M/2]= eveneven + weighted_evenodd - weighted_oddeven - weighted_oddodd
    ft[M/2:,M/2:]= eveneven - weighted_evenodd - weighted_oddeven + weighted_oddodd


    return ft/4 #Division by factor of 4 = 2 * 2 because of reverse Fourier transform, for each recursion step


def ift2_prec(qx):
    global W
    M=qx.rows
    N=qx.cols
    ft=mpm.zeros(M,N)

    #weight_range=mpm.matrix([mpm.exp(phase*mk) for mk in mpm.matrix(mpm.arange(M*M))/mpm.mpf(M)])
    for k in np.arange(qx.rows):
        for l in np.arange(qx.cols):
            for m in np.arange(qx.rows):
                for n in np.arange(qx.cols):
                    phase_fact= W[M,m*k] * W[M,n*l]
                    ft[k,l]+=qx[m,n]*phase_fact

    ft/=(M*N)
    return ft



def ift2_prec_4_at_a_time(qx1,qx2,qx3,qx4):
    global phase
    #All qx must have the same shape!!!!
    M=qx1.rows
    N=qx1.cols
    ft1=mpm.zeros(M,N)
    ft2=mpm.zeros(M,N)
    ft3=mpm.zeros(M,N)
    ft4=mpm.zeros(M,N)

    weight_range=mpm.matrix([mpm.exp(phase*mk) for mk in mpm.matrix(mpm.arange(M*M))/mpm.mpf(M)])

    for k in np.arange(qx1.rows):
        for l in np.arange(qx1.cols):
            for m in np.arange(qx1.rows):
                for n in np.arange(qx1.cols):
                    phase_fact= weight_range[m*k] * weight_range[n*l]
                    ft1[k,l]+=qx1[m,n]*phase_fact
                    ft2[k,l]+=qx2[m,n]*phase_fact
                    ft3[k,l]+=qx3[m,n]*phase_fact
                    ft4[k,l]+=qx4[m,n]*phase_fact

    ft1/=(M*N)
    ft2/=(M*N)
    ft3/=(M*N)
    ft4/=(M*N)
    return ft1, ft2, ft3, ft4


def piecewise_mat_mul_conj(A,B):
#could catch if shapes of A and B differ, todo.
    R=mpm.zeros(A.rows,A.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            R[i,j]=A[i,j]*mpm.conj(B[i,j])
    return R

def piecewise_mat_mul(A,B):
#could catch if shapes of A and B differ, todo.
    R=mpm.zeros(A.rows,A.cols)
    for i in range(A.rows):
        for j in range(A.cols):
            R[i,j]=A[i,j]*B[i,j]
    return R

def piecewise_mat_times_vec(A,v):
#could catch if shapes of A and B differ, todo.
    #assert v.rows==A.cols
    R=mpm.zeros(A.rows,A.cols)
    for i in range(v.rows):
        R[:,i]=A[:,i]*v[i]
    return R

def piecewise_vec_times_mat(v,A):
#could catch if shapes of A and B differ, todo.
    #assert v.cols==A.rows
    R=mpm.zeros(A.rows,A.cols)
    for i in range(v.cols):
        R[i,:]=A[i,:]*v[i]
    return R

def precalculate_W(N):
    global W
    phase=2*mpm.pi*mpm.mpc(0,"1")
    W=mpm.zeros(N+1,N*N)
    Nmut=N
    while Nmut%2==0:
        W[Nmut,:Nmut/2]=mpm.matrix([mpm.exp(phase*m) for m in mpm.matrix(mpm.arange(Nmut/2))/mpm.mpf(Nmut)]).T
        Nmut/=2

    W[Nmut,:Nmut*Nmut]=mpm.matrix([mpm.exp(phase*m) for m in mpm.matrix(mpm.arange(Nmut*Nmut))/mpm.mpf(Nmut)]).T #Full range!!


def cmatfull(M, bc, Lx, Ly):
    """
    M: parameter in the Chern insulator
    bc: Options: 'apbc' or 'pbc'. Flag that indicates the boundary conditions along the x-direction
    Lx: number of rows
    Ly: number of columns
    
    Boundary conditions along the y-direction assumed to be anti-periodic (this is to deal with the degeneracy at kx=ky=0)
    
    Returns the full correlation matrix"""
    
    
    #kylist = pi/Ly*arange(1,2*Ly+1,2)
    kylist = mpm.pi/Ly*mpm.matrix(mpm.arange(1,2*Ly+1,2),1)
    
    if bc=='pbc':
        #kxlist = 2*pi/Lx*arange(Lx)
        kxlist = 2*mpm.pi/Lx*mpm.matrix(mpm.arange(Lx),1)
    elif bc=='apbc':
        #kxlist = pi/Lx*arange(1,2*Lx+1,2)
        kxlist = mpm.pi/Lx*mpm.matrix(mpm.arange(1,2*Lx+1,2),1)
    else:
        raise Exception("The flag bc should be 'pbc' or 'apbc'")
    
    #print "Diagonalizing 2x2 k space matrices",time.clock()
    
    evecs = np.array([[one_chern_evec([M], kx, ky)[1] for ky in kylist] for kx in kxlist])
    evecs0 = mpm.matrix(evecs[:,:,0])
    evecs1 = mpm.matrix(evecs[:,:,1])
    #print "done, piecewise conj",time.clock()

    cdagcKspace=piecewise_mat_mul_conj(evecs0, (evecs0))
    ddagdKspace=piecewise_mat_mul_conj(evecs1, (evecs1))
    ddagcKspace=piecewise_mat_mul_conj(evecs0, (evecs1))
    cdagdKspace=piecewise_mat_mul_conj(evecs1, (evecs0))

    precalculate_W(cdagcKspace.rows)

    cdagc = ift2_prec_radix2_divide(cdagcKspace)
    ddagd = ift2_prec_radix2_divide(ddagdKspace)
    ddagc = ift2_prec_radix2_divide(ddagcKspace)
    cdagd = ift2_prec_radix2_divide(cdagdKspace)

    #print "done, ift2 ",time.clock()
    #cdagc,ddagd,ddagc,cdagd = ift2_prec_4_at_a_time(cdagcKspace,ddagdKspace,ddagcKspace,cdagdKspace)
    #cdagc,ddagd,ddagc,cdagd = ift2_prec_radix2_divide_4_at_a_time(cdagcKspace,ddagdKspace,ddagcKspace,cdagdKspace)

    return [[cdagc, cdagd], [ddagc, ddagd]]

def cmatA(maskA, corrfull):
    """
    maskA: mask with 1's in the position of A
    corrfull: the entire correlation matrix 
    
    Returns the correlation matrix on A"""
    
    (Lx, Ly) = maskA.shape
    (Arow, Acol) = np.nonzero(maskA)
    
    N = Arow.size
    
    CmatA = mpm.zeros(2*N)
    
    #Cmatfull = corrfull
    
    #CmatA[0:N, 0:N] = array([[Cmatfull[0][0][ mod(Arow[i]-Arow[j], Lx), mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    #CmatA[0:N, N:] = array([[Cmatfull[0][1][ mod(Arow[i]-Arow[j], Lx), mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    #CmatA[N:, 0:N] = array([[Cmatfull[1][0][ mod(Arow[i]-Arow[j], Lx), mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    #CmatA[N:, N:] = array([[Cmatfull[1][1][ mod(Arow[i]-Arow[j], Lx), mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    
    for i in range(N):
        for j in range(N):
            CmatA[i,j]=corrfull[0][0][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)]
            CmatA[i,j+N]=corrfull[0][1][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)]
            CmatA[i+N,j]=corrfull[1][0][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)]
            CmatA[i+N,j+N]=corrfull[1][1][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)]

    #print "CmatA", CmatA, CmatA.rows, CmatA.cols
    return CmatA


def ee_diffradii(shape, M, bc, epsx, epsy, Lmult, rad_all, alphas, fname):
    
    #If the radius is a number, not an array, make it an array
    #if len(rad_all)==1:
    #    rad_all = np.array(rad_all)
    
    rad_all = np.array(rad_all)
    
    #Append an empty comment line into the file 
    '''
    f = open(fname, 'a')
    f.write('# \n')
    f.close()
    '''
    
    L_all = np.array(rad_all*Lmult, dtype=int)
    
    for i in range(len(rad_all)):

        radius = rad_all[i]
        L = L_all[i]
        #Cfull = cmatfull_analytic(M, bc, L, L)
        #print "Initializations done",time.clock()
        Cfull = cmatfull(M, bc, L, L)
        print radius," ift done",time.clock()
        
        if shape=='circle':
            Lmid = int(L/2.)
            maskA = circle([Lmid+epsx, Lmid+epsy], radius, L, L)
        elif shape=='square':
            Lmid = int(L/2.)
            maskA = square([Lmid+epsx, Lmid+epsy], radius, L, L)
            
        
        #Compute the entanglement entropy of the region A
        corrA = cmatA(maskA, Cfull)

        #print "done, diagonalizing corrA",time.clock()
        pA = mpm.eighe(corrA, eigvals_only=True)
        repA=mpm.matrix([mpm.fabs(mpm.re(ev)) for ev in pA])
        print radius,"diagonalization done",time.clock()

        sent=mpm.zeros(len(alphas),1)

        for i,alpha in enumerate(alphas):
            if alpha==1:
                for j in range(repA.rows):
                    sent[i,0]+= -(repA[j]*mpm.log(repA[j]+eps)) - ((1-repA[j])*mpm.log(mpm.fabs(1-repA[j]+eps)))

            else:
                Renyi_pref=1/(1-alpha)
                for j in range(repA.rows):
                    sent[i,0]+= Renyi_pref * (mpm.log(mpm.power((repA[j]+eps),alpha) + mpm.power(mpm.fabs(1-repA[j]+eps),alpha)))

        
        #Appending the data to the end of the file with name fname
            f = open(fname[i], 'a')
            f.write('%f %.32f \n'%(radius, sent[i,0]))
            f.close()
    
        #print "done ",time.clock()
    return
    
