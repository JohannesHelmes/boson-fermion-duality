import numpy as np
import mpmath as mpm
#import time
#start_time = time.time()


###############################################################################################
########################################  getEntropy  #########################################
###############################################################################################
def getEntropy((Lx,Ly),alpha,regA,X,P):
    sitesA = []
    for x in range(Lx):
        for y in range(Ly):
            if regA[x,y] == True: sitesA.append(site((Lx,Ly),x,y))
    NsA=len(sitesA)

    Pred=mpm.matrix(len(sitesA),len(sitesA))
    Xred=mpm.matrix(len(sitesA),len(sitesA))
    for i,x in enumerate(sitesA):
        for j,y in enumerate(sitesA):
            Pred[i,j]=P[x,y]
            Xred[i,j]=X[x,y]

    Csquared = (Xred.T)*Pred


    #Ev = np.real(np.sqrt(np.linalg.eigvals(Csquared)))

    Ev = mpm.matrix(mpm.mp.eig(Csquared, left=False, right=False ) )



    reEv=mpm.matrix([mpm.sqrt(ev) for ev in Ev])
    EvPlusHalf=mpm.matrix([mpm.fabs(ev+0.5) for ev in reEv])
    EvMinusHalf=mpm.matrix([mpm.fabs(ev-0.5) for ev in reEv])

    #IS CHOPPING REALLY NECESSARY ??????????????????????????????????
    EvPlusHalf=mpm.chop(EvPlusHalf, tol=1e-32)
    EvMinusHalf=mpm.chop(EvMinusHalf, tol=1e-32)
    epsilon=mpm.mpf(1e-32)

#print np.linalg.eigvals(Csquared)
#print
#Sn = 0. 
    Sn = mpm.zeros(len(alpha),1)
    for j in range(0, NsA):
#if Ev[j] > 0.5:
        for i,n in enumerate(alpha):
            if n == 1:
                #Sn[i] += (Ev[j]+1./2)*np.log(abs(Ev[j]+1./2.)) - (Ev[j]-1./2.)*np.log(abs(Ev[j]-1./2))
                Sn[i,0] += (EvPlusHalf[j])*mpm.ln(EvPlusHalf[j]+epsilon) - (EvMinusHalf[j])*mpm.ln(EvMinusHalf[j]+epsilon)
            else:
                #Sn[i] += 1.0/(n-1.0) * np.log( (Ev[j]+1./2)**n - (Ev[j]-1./2.)**n )
                Sn[i,0] += 1.0/(n-1.0) * mpm.ln( mpm.power(EvPlusHalf[j],n) - mpm.power(EvMinusHalf[j],n ) )
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

def getRegionA((Lx,Ly),(x0,y0),bipart,angle=90):
    regA = np.zeros( (Lx,Ly), dtype='bool' )
    if angle==90:
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            regA[:,:y0]=True
        elif bipart==3:
            regA[x0:,y0:]=True
        else:
            regA[:x0,:y0]=True

    if angle==45:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        else:
            kathete_length=y0+x0
            for e,i in enumerate(range(max(Lx,kathete_length))):
                regA[e,0:max(Ly,kathete_length)]=True
            if bipart==3:
                regA[:x0,:]=False
                #take upper right part of bipart 2
            elif bipart==4:
                regA=np.logical_not(regA)
                regA[x0:,:]=False

    #print regA

    if( (regA==True).sum() > (Lx*Ly/2) ): 
        regA = np.logical_not(regA)

    return regA

###############################################################################################
###########################################  site  ############################################
###############################################################################################
def site((Lx,Ly),x,y):
  return x + (y*Lx)  #convert (x,y) pair to a single site number
