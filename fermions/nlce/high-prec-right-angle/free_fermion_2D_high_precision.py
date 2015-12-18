import numpy as np
import pprint
import mpmath as mpm

###############################################################################################
########################################  getEntropy  #########################################
###############################################################################################
def getEntropy((Lx,Ly),alpha,regA,Hamiltonian):
    sitesA = []
    for x in range(2*Lx):
        for y in range(Ly):
            if regA[x,y] == True: sitesA.append(site((2*Lx,Ly),x,y))
    NsA=len(sitesA)
    #Probably np.flatten does it
    sitesA.sort()

    #print Hamiltonian

    Eval,Evec = mpm.mp.eighe(Hamiltonian) 
    #print Eval


    U=mpm.matrix(Evec)

    #print U


    '''
    GSOccup=np.kron(([1,0],[0,0]),np.eye(Lx*Ly))
    C= U*GSOccup*U.getH()
    '''

    GSU=U[:,0:Lx*Ly]

    GSU_star=mpm.matrix(GSU.rows,GSU.cols)
    for y in range(GSU.cols):
        for x in range(GSU.rows):
            GSU_star[x,y]=mpm.conj(GSU[x,y])

    C=GSU*GSU_star.T


    #Cred=C[np.ix_(sitesA,sitesA)]

    Cred=mpm.matrix(len(sitesA),len(sitesA))
    for i,x in enumerate(sitesA):
        for j,y in enumerate(sitesA):
            Cred[i,j]=C[x,y]

    Ev = (mpm.mp.eig(Cred, left=False, right=False ) )
    reEv=mpm.matrix([mpm.re(ev) for ev in Ev])
    OneMinusEv=mpm.matrix([mpm.fabs(1.-ev) for ev in reEv])
    #Ev_float=np.linalg.eigvals(Cred)
    reEv=mpm.chop(reEv, tol=1e-24)
    OneMinusEv=mpm.chop(OneMinusEv, tol=1e-24)
    epsilon=mpm.mpf(1e-24)
    #print Ev
    #print Ev_float

    #Sn = mpm.matrix(np.zeros(len(alpha)))
    Sn = np.zeros(len(alpha))
#Sn = 0. 
    
    for j in range(0, NsA):
        #print reEv[j]
        for i,n in enumerate(alpha):
            if n==1:
                Sn[i]+= - reEv[j]*mpm.ln(reEv[j]+epsilon) - (OneMinusEv[j])*mpm.ln(OneMinusEv[j]+epsilon)
            else:
                try:
                    Sn[i]+=(1./(1.-n))*mpm.ln(mpm.power(reEv[j],n) + mpm.power(OneMinusEv[j],n))
                except TypeError:
                    print "Type Error", reEv[j], "^", n, "=", mpm.power(reEv[j]+epsilon,n), " and ", 1-reEv[j], "^", n, "=", mpm.power(1-reEv[j]+epsilon,n)


    '''
    #RenyiSums=np.ones((1))
    vonNeumannS=mpm.mpf('0.0')
    for j in range(0, NsA):
#if Ev[j] > 0.5:
        #if (mpm.re(Ev[j])  > 1e-12)and(1-mpm.re(Ev[j])> 1e-12): #Numerical security against floating point roundoff errors
        if (mpm.re(Ev[j])  > 0)and(1-mpm.re(Ev[j])> 0): #Numerical security against floating point roundoff errors
            vonNeumannS += - (mpm.re(Ev[j]))*mpm.ln(mpm.re(Ev[j])) - (1.-mpm.re(Ev[j]))*mpm.ln(1.-mpm.re(Ev[j]))
        #RenyiSums=np.append(Ev[j].real*RenyiSums,(1-Ev[j].real)*RenyiSums)
            for i,n in enumerate(alpha):
                if n!=1:
                    Sn[i]+=mpm.ln(mpm.power(mpm.re(Ev[j]),n) + mpm.power((1-mpm.re(Ev[j])),n))

    for i,n in enumerate(alpha):
        if n==1:
            Sn[i]=vonNeumannS
        else:
            Sn[i]=(1./(1.-n))*(Sn[i])

    #print Sn
    '''
    return Sn
#........................................END getEntropy........................................

###############################################################################################
########################################  getRegionA  #########################################
###############################################################################################

def getRegionA((Lx,Ly),(x0,y0),bipart): #all factors of two due to fermion spin 
    regA = np.zeros( (2*Lx,Ly), dtype='bool' ) #doubling in x direction only!
    if bipart==1:
        regA[:2*x0,:]=True
    elif bipart==2:
        regA[:,:y0]=True
    elif bipart==3:
        regA[2*x0:,y0:]=True
    else:
        regA[:2*x0,:y0]=True

    if( (regA==True).sum() > (2*Lx*Ly/2) ): 
        regA = np.logical_not(regA)
    
    #print regA

    return regA

###############################################################################################
###########################################  site  ############################################
###############################################################################################
def site((Lx,Ly),x,y):
  return x + (y*Lx)  #convert (x,y) pair to a single site number
