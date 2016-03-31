#!/usr/bin/env python
# encoding: utf-8
"""
EntanglementFuncs_Haldane.py

Created by Anushya Chandran on 2015-06-30.

"""


#from pylab import *
import os
import numpy as np
from cmath import phase
from scipy.optimize import root


def one_chern_evec(Hamargs, kx, ky, chkspec= False):
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
  d = np.array([np.sin(kx) , np.sin(ky), 2 + M - np.cos(kx) - np.cos(ky)])
  
  sigx = np.array([ [0,1], [1,0] ])
  sigy = np.array([ [0, -1j], [ 1j,0] ], dtype = 'complex')
  sigz = np.array([ [1, 0], [0, -1]])
  
  darr = d[0]*sigx + d[1]*sigy + d[2]*sigz

  (en, evecs) = np.linalg.eigh(darr)

  
  if chkspec:
    return en
  else:
    return (en[0], evecs[:,0])


def circle(center_coord, rad, Lx, Ly):
    """
    center_coord: tuple with (row, col) index of the center of the circle
    rad: Radius
    Lx: total number of rows
    Ly: total number of columns
    
    Returns a Lx X Ly array with 1s where the circle is.
    """
    
    fracy = center_coord[1] - np.floor(center_coord[1])
    fracx = center_coord[0] - np.floor(center_coord[0])
    
    #First the canonical circle with center coordinates (Lx/2+fracx, Ly/2+fracy)
    (cols, rows) = meshgrid( arange(Ly),arange(Lx))
    
    centerx = int(Lx/2.)
    centery = int(Ly/2.)
    
    distances = sqrt( (cols - ( centery +fracy ) )**2 + (rows - ( centerx+fracx ) )**2)
    
    mask = distances<rad
    
    tmpmask = np.roll(mask, int(np.floor(center_coord[1])) - centery, axis=1)
    return np.roll(tmpmask, int(np.floor(center_coord[0])) - centerx, axis=0)    

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
    

def triangle(center_coord, rad, Lx, Ly):
    """
    creates a triangle with one 90 degree and to 45 degree corners. The 90 corner is upper left.
    The center_coord is in the middle of the hypothenuse
    center_coord: tuple with (row, col) index of the center of the square
    rad: Length of the side of the square
    Lx: total number of rows
    Ly: total number of columns
    
    Returns a Lx X Ly array with 1s where the triangle is.
    """
    
    edgex = int(Lx/2.) - int(rad/2)
    edgey = int(Ly/2.) - int(rad/2)
    
    mask = np.zeros((Lx,Ly), dtype='bool')

    for i,e in enumerate(range(edgex,edgex+int(rad))):
        mask[e,edgey:edgey + int(rad) - i] = True
    #mask[edgex:edgex + int(rad), edgey:edgey + int(rad)] = True
    
    centerx = int(Lx/2.)
    centery = int(Ly/2.)
        
    tmpmask = np.roll(mask, int(np.floor(center_coord[1])) - centery, axis=1)
    return np.roll(tmpmask, int(np.floor(center_coord[0])) - centerx, axis=0)


def octagon(center_coord, rad, Lx, Ly):
    """
    creates an octagon with three times the radius as its width and height
    has 8 135 degree corners
    center_coord: tuple with (row, col) index of the center of the square
    rad: Length of the side of the square
    Lx: total number of rows
    Ly: total number of columns
    
    Returns a Lx X Ly array with 1s where the octagon is.
    """
    
    edgex = int(Lx/2.) - int(rad/2)
    edgey = int(Ly/2.) - int(rad/2)
    
    mask = np.zeros((Lx,Ly), dtype='bool')


    mask[edgex-int(rad):edgex + 2*int(rad), edgey-int(rad):edgey + 2*int(rad)] = True
    for i,e in enumerate(range(edgex-int(rad),edgex)):
        mask[e,edgey-int(rad):edgey - i] = False
        mask[e,edgey+2*int(rad):edgey+int(rad) + i-1 :-1] = False

    for i,e in enumerate(range(edgex+int(rad),edgex+2*int(rad))):
        mask[e,edgey-int(rad):edgey -int(rad) + i + 1] = False
        mask[e,edgey+2*int(rad):edgey +2*int(rad) - i -2 :-1] = False
    #mask[edgex:edgex + int(rad), edgey:edgey + int(rad)] = True
    
    centerx = int(Lx/2.)
    centery = int(Ly/2.)
        
    tmpmask = np.roll(mask, int(np.floor(center_coord[1])) - centery, axis=1)
    return np.roll(tmpmask, int(np.floor(center_coord[0])) - centerx, axis=0)


def twosquares(rad, distance, Lx, Ly):
    """rad: length of each square
    distance: distance between the two squares measured from their centers
    Lx: length of the sample in the x-direction
    Ly: length of the sample in the y-direction
    
    Returns three Lx X Ly arrays.
    The first is where the first square is, the second is where the second sqaure is, and third is where the two squares are. The two squares are dispaced by distance in the x-direction."""
    
    cx = int(Lx/2.)
    cy = int(Ly/2.)
    
    x1 = cx + int( (distance)/2)
    
    x2 =  x1 - distance
    
    m1 = square(( x1, cy), rad, Lx, Ly)
    m2 = square(( x2, cy), rad, Lx, Ly)
    
    return (m1, m2, logical_or(m1,m2))
    
def phi(kx, ky):
    return phase(complex(np.sin(kx), np.sin(ky) ) )

def vec_func1(k, mn, Lx, Ly):
    kx, ky = k
    m, n = mn

    cond_x = kx * (Lx + 1) - m * np.pi - phi(ky, kx)
    cond_y = ky * (Ly + 1) - n * np.pi - phi(kx, ky)

    return (cond_x, cond_y)

def generate_klists(bc, Lx, Ly):
    """
    generates all k values depending on the boundary conditions (pbc, apbc, obc)
    """
    
    if bc=='pbc':
        kylist = np.pi/Ly*np.arange(1,2*Ly+1,2)
        kxlist = 2*np.pi/Lx*np.arange(Lx)
        kxgrid, kygrid = np.meshgrid(kxlist, kylist)
        for n in xrange(Ly):
            for m in xrange(Lx):
                kxgrid[m,n] = kxlist[m]
                kygrid[m,n] = kylist[n]

    elif bc=='apbc':
        kylist = np.pi/Ly*np.arange(1,2*Ly+1,2)
        kxlist = np.pi/Lx*np.arange(1,2*Lx+1,2)
        kxgrid, kygrid = np.meshgrid(kxlist, kylist)
    elif bc=='obc':
        tol=1e-12
        kxlist=[]
        kylist=[]
        kxgrid = np.zeros((Lx,Ly))
        kygrid = np.zeros((Lx,Ly))
        for n in xrange(1,Ly+1):
            for m in xrange(1,Lx+1):
                Kx, Ky = root(vec_func1,(np.pi/2, np.pi/2), args=([m,n], Lx, Ly) ).x
                kxgrid[m-1,n-1] = Kx
                kygrid[m-1,n-1] = Ky

    else:
        raise Exception("The flag bc should be 'pbc' or 'apbc'")

    return kxgrid, kygrid
    
def cmatfull_analytic(M, bc, Lx, Ly):
    
    if bc=='pbc':
        kylist = np.pi/Ly*np.arange(1,2*Ly+1,2)
        kxlist = 2*np.pi/Lx*np.arange(Lx)
    elif bc=='apbc':
        kylist = np.pi/Ly*np.arange(1,2*Ly+1,2)
        kxlist = np.pi/Lx*np.arange(1,2*Lx+1,2)
        
    kxgrid, kygrid = np.meshgrid(kxlist, kylist)
    
    dx = sin(kxgrid)
    dy = sin(kygrid)
    dz = 2 + M - cos(kxgrid) - cos(kygrid)
    
    modd = sqrt(dx**2 + dy**2 + dz**2)
    
    cdagc = ifft2(- 0.5*dz/modd)
    cdagc[0,0] += 0.5
    ddagd = -cdagc
    ddagd[0,0] = 1.0 - cdagc[0,0]
    
    
    tx = 1j * imag(-1*ifft2( dx/(2*modd) ))
    ty = 1j * imag(-1*ifft2( dy/(2*modd) ))
    
    cdagd = -1*ifft2( (dx + 1j*dy)/(2*modd) )
    ddagc = -1*ifft2( (dx - 1j*dy)/(2*modd) )
    
    #cdagd = tx + 1j*ty
    #ddagc = tx -1j*ty
    
    return [[cdagc, cdagd], [ddagc, ddagd]]
    
def ift2_prec(qx):
    M,N=qx.shape
    ft=np.zeros((M,N),dtype='complex')

    phase=2*np.pi*1j
    for k in np.arange(M):
        for l in np.arange(N):
            for m in nparange(M):
                for n in np.arange(N):
                    weight=float(m*k)/M + float(n*l)/N
                    ft[k,l]+=qx[m,n]*np.exp(phase*weight)

    ft/=(M*N)
    return ft

def cmatfull(M, bc, Lx, Ly):
    """
    M: parameter in the Chern insulator
    bc: Options: 'apbc' or 'pbc'. Flag that indicates the boundary conditions along the x-direction
    Lx: number of rows
    Ly: number of columns
    
    Boundary conditions along the y-direction assumed to be anti-periodic (this is to deal with the degeneracy at kx=ky=0)
    
    Returns the full correlation matrix"""
    
    
    kxgrid, kygrid = generate_klists(bc, Lx, Ly)
    
    evecs = np.array([[one_chern_evec([M], kx, ky)[1] for (kx, ky) in zip(rowx, rowy)] for (rowx, rowy) in zip(kxgrid[:],kygrid[:])])
    en = np.array([[one_chern_evec([M], kx, ky)[0] for (kx, ky) in zip(rowx, rowy)] for (rowx, rowy) in zip(kxgrid[:],kygrid[:])])

    print en

    cdagc = np.fft.ifft2(evecs[:,:,0] * np.conj(evecs[:,:,0]))
    ddagd = np.fft.ifft2(evecs[:,:,1] * np.conj(evecs[:,:,1]))
    
    ddagc = np.fft.ifft2(evecs[:,:,0] * np.conj(evecs[:,:,1]))
    cdagd = np.fft.ifft2(evecs[:,:,1] * np.conj(evecs[:,:,0]))

    return [[cdagc, cdagd], [ddagc, ddagd]]

def cmatA(maskA, corrfull):
    """
    maskA: mask with 1's in the position of A
    corrfull: the entire correlation matrix 
    
    Returns the correlation matrix on A"""
    
    (Lx, Ly) = maskA.shape
    (Arow, Acol) = np.nonzero(maskA)
    
    N = Arow.size
    
    CmatA = np.zeros((2*N, 2*N), dtype='complex')
    
    Cmatfull = corrfull
    
    CmatA[0:N, 0:N] = np.array([[Cmatfull[0][0][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    CmatA[0:N, N:] = np.array([[Cmatfull[0][1][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    CmatA[N:, 0:N] = np.array([[Cmatfull[1][0][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    CmatA[N:, N:] = np.array([[Cmatfull[1][1][ np.mod(Arow[i]-Arow[j], Lx), np.mod(Acol[i]-Acol[j], Ly)] for j in range(N)] for i in range(N)])
    
    #print "CmatA", CmatA, CmatA.shape
    return CmatA

def mutinf_annulus_diffradii(M, bc, epsx, epsy, width, Lmult, rad_all, fname):
    
    #If the radius is a number, not a list, make it a list
    if np.isscalar(rad_all):
        rad_all = array([rad_all])
    
    #Append an empty comment line into the file 
    f = open(fname, 'a')
    f.write('# \n')
    f.close()
    
    L_all = array(rad_all*Lmult, dtype=int)
    sent = zeros(3)
    
    for i in range(len(rad_all)):

        radius = rad_all[i]
        L = L_all[i]
        Cfull = cmatfull_analytic(M, bc, L, L)

        rminus = radius - width/2.
        rplus = radius + width/2.

        Lmid = int(L/2.)
        maskAminus = circle([Lmid+epsx, Lmid+epsy], rminus, L, L)
        maskAplus = circle([Lmid+epsx, Lmid+epsy], rplus, L, L)

        maskAstrip = np.logical_xor(maskAminus, maskAplus)

        #Inner circle
        corrA = cmatA(maskAminus, Cfull)
        pA = real(np.linalg.eigvalsh(corrA))
        sent[0] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))

        #Outer circle
        corrA = cmatA(maskAplus, Cfull)
        pA = real(np.linalg.eigvalsh(corrA))
        sent[1] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))

        #Strip
        corrA = cmatA(maskAstrip, Cfull)
        pA = real(np.linalg.eigvalsh(corrA))
        sent[2] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))
        
        #Appending the data to the end of the file with name fname
        f = open(fname, 'a')
        f.write('%f %f %f %f \n'%(radius, sent[0], sent[1], sent[2]))
        f.close()
    
    return

def ee_diffradii(shape, M, bc, epsx, epsy, L_all, rad_all, alphas, fname):
    
    #If the radius is a number, not an array, make it an array
    if np.isscalar(rad_all):
        rad_all = np.array([rad_all])
    
    rad_all = np.array(rad_all)
    
    #Append an empty comment line into the file 
    '''
    f = open(fname, 'a')
    f.write('# \n')
    f.close()
    '''
    
    
    for i in range(len(rad_all)):

        radius = rad_all[i]
        L = L_all[i]
        #Cfull = cmatfull_analytic(M, bc, L, L)
        Cfull = cmatfull(M, bc, L, L)
        
        if shape=='circle':
            Lmid = int(L/2.)
            maskA = circle([Lmid+epsx, Lmid+epsy], radius, L, L)
        elif shape=='square':
            Lmid = int(L/2.)
            maskA = square([Lmid+epsx, Lmid+epsy], radius, L, L)
        elif shape=='triangle':
            Lmid = int(L/2.)
            maskA = triangle([Lmid+epsx, Lmid+epsy], radius, L, L)
        elif shape=='octagon':
            Lmid = int(L/2.)
            maskA = octagon([Lmid+epsx, Lmid+epsy], radius, L, L)
            
        
        #Compute the entanglement entropy of the region A
        corrA = cmatA(maskA, Cfull)
        #print corrA
        pA = abs(np.real(np.linalg.eigvalsh(corrA)))

        #print pA
        sent=np.zeros(len(alphas))
        for i,alpha in enumerate(alphas):
            if alpha==1:
                sent[i] = -1* sum(pA*np.log(pA+1e-15))+ -1* sum((1-pA)*np.log(abs(1-pA+1e-15)))
            else:
                sent[i] = 1./(1.-alpha) * sum(np.log((pA+1e-15)**alpha + abs(1-pA+1e-15)**alpha))

        
        #Appending the data to the end of the file with name fname
            f = open(fname[i], 'a')
            f.write('%f %f %f \n'%(L, radius, sent[i]))
            f.close()
    
    return


def mutinf_twosquares_diffradii(M, bc, squarelen, L, dist_all):
    
    
    fname = 'MI_square_M%1.2f_squarelen%1.2f_L%1.2f_'%(M, squarelen, L)+bc+'.txt'
   
    f = open( fname, 'w')
    
    #Write all the parameters of the run into the file as comments
    f.write('# M=%f\n'%M)
    f.write('# SquareLength=%f\n'%squarelen)
    f.write('# L=%f\n'%L)
    f.write('# bc='+bc+'\n')
    f.write('# shape=square\n')
    
    
    #If the distance is a number, make it an array
    if np.isscalar(dist_all):
        dist_all = np.array([dist_all])
    
    #Append an empty comment line into the file 
    f = open(fname, 'a')
    f.write('# \n')
    f.close()
    
    #Compute the correlation matrix in the entire system
    Cfull = cmatfull_analytic(M, bc, L, L)
    
    sent = zeros(3)
    
    for i in range(len(dist_all)):

        dist = dist_all[i]
        
        (m1, m2, m12) = twosquares(squarelen, dist, L, L)
        
        #As the individual squares have the same entropy, we compute their entanglement entropy only once
        if i==0:
            #First square
            corrA = cmatA(m1, Cfull)
            pA = real(eigvalsh(corrA))
            sent[0] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))

            #Second square
            corrA = cmatA(m2, Cfull)
            pA = real(eigvalsh(corrA))
            sent[1] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))

        #Both squares
        corrA = cmatA(m12, Cfull)
        pA = real(eigvalsh(corrA))
        sent[2] = -1* sum(pA*log2(pA+1e-10))+ -1* sum((1-pA)*log2(1-pA+1e-10))
        
        #Appending the data to the end of the file with name fname
        f = open(fname, 'a')
        f.write('%f %f %f %f \n'%(dist, sent[0], sent[1], sent[2]))
        f.close()
    
    return

def momentumblock_cmatA(M, kpar, Lperp, La):
    
    """M: Parameter of the Chern Hamiltonian 
    kpar: Momentum along the length of the cut
    Lperp: Length of the system perperpendicular to the cut
    La: Length of part A
    Returns a one-dimensional correlation matrix in the ground state of the lattice model."""
    
    evecs = np.array([one_chern_evec( [M], kx, kpar)[1] for kx in arange(Lperp)*2*pi/Lperp])
    sub = 2
    lowbands = 1
    Crow = [[ifft( conj( evecs[:,i] ) * evecs[:,j] ) for i in range(sub)] for j in range(sub)]
    
    Ctranslate = [[np.array([[Crow[i][j][mod(n-m, Lperp)]  for m in range(La)] for n in range(La)]) for i in range(sub)] for j in range(sub)]
  
    Cmat = zeros((sub*La, sub*La), dtype='c16')
  
    for Bi in range(sub):
      for Bj in range(sub):
        Cmat[Bi*La:(Bi+1)*La, Bj*La:(Bj+1)*La] = Ctranslate[Bi][Bj]
  
    return Cmat



def unittest_circley():
    """This function is a unit test. It checks that entanglement spectrum (eigenvalues and eigenvectors) obtained from 
    the cmatA function agree with the analytical expectation"""
    
    L = 20
    M = 1.2
    
    
    kyall = pi/L*arange(1,2*L+1,2)
    
    ee_specs = zeros(2*L)
    
    for i in range(L):
        
        ky = kyall[i]
        evecs = np.array([one_chern_evec( [M], kx, ky)[1] for kx in 2*pi/L*arange(L)])
        
        cmat = zeros((2,2), dtype='complex')
        
        cmat[0,0] = evecs[:,0].conj().dot(evecs[:,0])/L
        cmat[0,1] = evecs[:,1].conj().dot(evecs[:,0])/L
        cmat[1,0] = evecs[:,0].conj().dot(evecs[:,1])/L
        cmat[1,1] = evecs[:,1].conj().dot(evecs[:,1])/L
        
        pvals = real(eigvalsh(cmat))
        
        ee_specs[2*i:2*i+2] = pvals
    
    maskA = zeros((L,L))
    maskA[5,:] = ones(L)
    corrfull = cmatfull(M, 'pbc', L, L)
    cA = cmatA(maskA, corrfull)
    
    ee_specs2 = eigvalsh(cA)
    
    if not( allclose(ee_specs2, sort(ee_specs) ) ):
        raise Exception("The entanglement spectrum obtained by using momentum does not agree with that produced by cmatA using cmatfull to construct the entire correlation matrix")
    
    corrfull = cmatfull_analytic(M, 'pbc', L, L)
    cA = cmatA(maskA, corrfull)
    
    ee_specs2 = eigvalsh(cA)
    
    if not( allclose(ee_specs2, sort(ee_specs) ) ):
        raise Exception("The entanglement spectrum obtained by using momentum does not agree with that produced by cmatA using cmatfull_analytic to construct the entire correlation matrix")
    
    return
    
        

def unittest_one_chern_evec():
    """This unittest checks that one_chern_evec produces the right eigenvectors"""
    
    M = -1.3
    
    kx = 2*pi*rand()
    ky = 2*pi*rand()
    
    d = np.array([sin(kx) , sin(ky), 2 + M - cos(kx) - cos(ky)])
    
    dmat = array([ [d[2], d[0]-1j*d[1]], [d[0]+1j*d[1], -d[2]]]  )
    (e1, evec1) = eigh(dmat)
    
    (e2, evec2) = one_chern_evec( [M], kx, ky)
    
    if not(isclose(e1[0], e2)):
        print e1[0], e2
        raise Exception("The energy is incorrect at kx=%1.2f, ky=%1.2f"%(kx,ky))
    
    olap = evec1[:,0].conj().dot(evec2)
    
    if not( isclose(abs(olap), 1) ):
        print olap
        raise Exception("The eigenvector is incorrect at kx=%1.2f, ky=%1.2f becuase abs(olap) differs from one"%(kx,ky))
        
    if not(allclose(evec1[:,0], 1./olap*evec2)):
        print evec1[:,0], 1./olap*evec2
        raise Exception("The eigenvector is incorrect at kx=%1.2f, ky=%1.2f"%(kx,ky))
    return

def unittest_square():
    """This unittest checks that square function produces the correct masks"""
    
    Lx = 5
    Ly = 4
    
    maskchk = zeros((Lx,Ly), dtype=bool)
    maskchk[1,3] = True
    
    if not(all(square([1.9, 3.1], 1, Lx, Ly) == maskchk)):
        raise Exception("The square mask function does not produce the right mask of length 1")
        
    maskchk = zeros((Lx,Ly), dtype=bool)
    maskchk[1,1] = True
    maskchk[1,2] = True
    maskchk[2,1] = True
    maskchk[2,2] = True
    
    if not(all(square([2.5, 2.5], 2.1, Lx, Ly) == maskchk)):
        raise Exception("The square mask function does not produce the right mask of length 2")
    
    return
    
    
