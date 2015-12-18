import numpy as np
import scipy.linalg as la
#import time
#start_time = time.time()

"""
###############################################################################################
#######################################  getCornerEnt  ########################################
###############################################################################################
def getCornerEnt(Lx,Ly,alpha,massterm):
    perx = pery = False  #PBC or not along the x, y and z directions

    Ns = Lx * Ly
    L  = (Lx,Ly)

    '''

    K = np.zeros((Ns, Ns))
#Loop to calculate the matrix K:
    
    for x in range(Lx):
        for y in range(Ly):
            K[site(L,x,y),site(L,x,y)]= 4.0 + (massterm ** (2))
            xp = (x+1)%Lx
            yp = (y+1)%Ly
            if (xp > x) or perx:
                K[site(L,xp,y), site(L,x, y)] = -1.0 
                K[site(L,x, y), site(L,xp,y)] = -1.0
            if (yp > y) or pery:
                K[site(L,x,yp), site(L,x,y)] = -1.0 
                K[site(L,x,y), site(L,x,yp)] = -1.0

    Kold=K[:]
    '''


    K=(4.0 + massterm**2)*np.eye(Ns)+np.kron(np.eye(Ly),np.diag([-1.0]*(Lx-1),1)+np.diag([-1.0]*(Lx-1),-1))
    if Lx>0:
        K+=np.diag([-1.]*(Ns-Lx),Lx)+np.diag([-1.]*(Ns-Lx),-Lx)

    #Eval,Evec = np.linalg.eigh(K) #, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)
#print Evec
    #EvMat=np.matrix(Evec)
    #EvaSqrt=np.sqrt(Eval)
    SqrtK=la.sqrtm(K)
    P = 1./2. * SqrtK
    X = 1./2. * la.inv(SqrtK)

    #P = 1./2. * EvMat * np.diag(EvaSqrt) * EvMat.T
    #X = 1./2. * EvMat * np.diag(1. /EvaSqrt) * EvMat.T

#result=0
    result = np.zeros(len(alpha))
#loop over all possible locations of the corner:
    horEntros= np.zeros((Ly/2+1,len(alpha)))
    verEntros= np.zeros((Lx,len(alpha)))
    
    for y0 in range(1,Ly/2+1):
        horEntros[y0]= getEntropy(L,alpha,getRegionA(L,(0,y0),2),X,P)
    
    for x0 in range(1,Lx):
        verEntros[x0]= getEntropy(L,alpha,getRegionA(L,(x0,0),1),X,P)
        
    for y0 in range(1,(Ly+1)/2):
        for x0 in range(1,Lx):
            r0 = (x0,y0) #position of the corner
            #corners:
            result+= (getEntropy(L,alpha,getRegionA(L,r0,4),X,P)
                             + getEntropy(L,alpha,getRegionA(L,r0,3),X,P)
                             - horEntros[y0]
                             - verEntros[x0])
    if Ly%2==0:
        y0=Ly/2
        maxX=(Lx+1)/2
        for x0 in range(1,maxX):
            r0 = (x0,y0) #position of the corner
            #corners:
            result+= (getEntropy(L,alpha,getRegionA(L,r0,4),X,P)
                             + getEntropy(L,alpha,getRegionA(L,r0,3),X,P)
                             - horEntros[y0]
                             - verEntros[x0])
        x0=Lx/2
        y0=Ly/2
        if (Lx%2==0)and(Ly%2==0):
            r0 = (x0,y0) #position of the corner
            #corners:
            result+= 1./2.*(getEntropy(L,alpha,getRegionA(L,r0,4),X,P)
                             + getEntropy(L,alpha,getRegionA(L,r0,3),X,P)
                             - horEntros[y0]
                             - verEntros[x0])

    return result
#.......................................END getCornerEnt.......................................
"""

###############################################################################################
########################################  getEntropy  #########################################
###############################################################################################
def getEntropy((Lx,Ly),alpha,regA,X,P):
    sitesA = []
    for x in range(Lx):
        for y in range(Ly):
            if regA[x,y] == True: sitesA.append(site((Lx,Ly),x,y))
    NsA=len(sitesA)

    Pred=P[np.ix_(sitesA,sitesA)]
    Xred=X[np.ix_(sitesA,sitesA)]

    Csquared = (Xred.T).dot(Pred)
    Ev = np.real(np.sqrt(np.linalg.eigvals(Csquared)))
#print np.linalg.eigvals(Csquared)
#print
#Sn = 0. 
    Sn = np.zeros(len(alpha))
    for j in range(0, NsA):
#if Ev[j] > 0.5:
        if Ev[j].real > 0.5: #Numerical security against floating point roundoff errors
            for i,n in enumerate(alpha):
                if n == 1:
                    Sn[i] += (Ev[j]+1./2)*np.log(abs(Ev[j]+1./2.)) - (Ev[j]-1./2.)*np.log(abs(Ev[j]-1./2))
                else:
                    Sn[i] += 1.0/(n-1.0) * np.log( (Ev[j]+1./2)**n - (Ev[j]-1./2.)**n )
    return Sn
#........................................END getEntropy........................................

###############################################################################################
########################################  getRegionA  #########################################
###############################################################################################
def getRegionAold((Lx,Ly),(x0,y0),fx,fy):
    regA = np.zeros( (Lx,Ly), dtype='bool' )

    for x in range(Lx):
        for y in range(Ly):
            if (fx(x,x0) and fy(y,y0)):
                regA[x,y] = True

#if the size of region A is more than half of the lattice, swap regions A and B:
    if( (regA==True).sum() > (Lx*Ly/2) ): regA = np.logical_not(regA)

    return regA

def getRegionA((Lx,Ly),(x0,y0),bipart):
    regA = np.zeros( (Lx,Ly), dtype='bool' )
    if bipart==1:
        regA[:x0,:]=True
    elif bipart==2:
        regA[:,:y0]=True
    elif bipart==3:
        regA[x0:,y0:]=True
    else:
        regA[:x0,:y0]=True

    if( (regA==True).sum() > (Lx*Ly/2) ): 
        regA = np.logical_not(regA)

    return regA

###############################################################################################
###########################################  site  ############################################
###############################################################################################
def site((Lx,Ly),x,y):
  return x + (y*Lx)  #convert (x,y) pair to a single site number
