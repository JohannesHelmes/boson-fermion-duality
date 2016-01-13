import numpy as np
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

    if angle==22.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(x0-2,-1,-1)):
                regA[e,0:min(Ly,y0+2*i+2)]=True

            regA[x0-1,0:min(Ly,y0)]=True
            for i,e in enumerate(range(x0,Lx)):
                regA[e,0:max(0,y0-2*i)]=True
        elif bipart==3:
            #regA[x0-1,y0:Ly]=True
            for i,e in enumerate(range(x0-1,-1,-1)):
                regA[e,min(y0+2*i,Ly):]=True

    if angle==45:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(x0-1,-1,-1)):
                regA[e,0:min(Ly,y0+i+1)]=True

            for i,e in enumerate(range(x0,Lx,1)):
                regA[e,0:max(0,y0-i)]=True
            #concave corners of the pixelized line are the locations of the (x0,y0)
        elif bipart==3:
            for i,e in enumerate(range(x0-1,-1,-1)):
                regA[e,min(y0+i,Ly):]=True

    if angle==67.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(x0-2,-1,-2)):
                regA[e,0:min(Ly,y0+i+1)]=True
                if e>0:
                    regA[e-1,0:min(Ly,y0+i+1)]=True

            regA[x0,0:min(Ly,y0)]=True
            regA[x0-1,0:min(Ly,y0)]=True
            for i,e in enumerate(range(x0+1,Lx,2)):
                regA[e,0:max(0,y0-i-1)]=True
                if e<Lx-1:
                    regA[e+1,0:max(0,y0-i-1)]=True
        elif bipart==3:
            regA[x0-1,y0:Ly]=True
            for i,e in enumerate(range(x0-2,-1,-2)):
                regA[e,min(y0+i+1,Ly):]=True
                if e>0:
                    regA[e-1,min(y0+i+1,Ly):]=True

    if angle==112.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(x0-2,-1,-2)):
                regA[e,0:min(Ly,y0+i+1)]=True
                if e>0:
                    regA[e-1,0:min(Ly,y0+i+1)]=True

            regA[x0,0:min(Ly,y0)]=True
            regA[x0-1,0:min(Ly,y0)]=True
            for i,e in enumerate(range(x0+1,Lx,2)):
                regA[e,0:max(0,y0-i-1)]=True
                if e<Lx-1:
                    regA[e+1,0:max(0,y0-i-1)]=True
        elif bipart==3:
            regA[x0-1,y0:Ly]=True
            for i,e in enumerate(range(x0-2,-1,-2)):
                regA[e,max(y0-i-1,0):]=True
                if e>0:
                    regA[e-1,max(y0-i-1,0):]=True


    if angle==135:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            #concave corners of the pixelized line are the locations of the (x0,y0)
            triangle_length=y0+x0-1
            for i,e in enumerate(range(min(Lx,triangle_length))):
                regA[e,0:min(Ly,triangle_length-i)]=True
        elif bipart==3:
            for i,e in enumerate(range(x0,0,-1)):
                regA[e-1,max(0,y0-i):Ly]=True

    if angle==157.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(x0-2,-1,-1)):
                regA[e,0:min(Ly,y0+2*i+2)]=True

            regA[x0-1,0:min(Ly,y0)]=True
            for i,e in enumerate(range(x0,Lx)):
                regA[e,0:max(0,y0-2*i)]=True
        elif bipart==3:
            #regA[x0-1,y0:Ly]=True
            for i,e in enumerate(range(x0-1,-1,-1)):
                regA[e,max(y0-2*i,0):]=True



    #print regA.T
            
    if( (regA==True).sum() > (Lx*Ly/2) ): 
        regA = np.logical_not(regA)

    return regA

###############################################################################################
###########################################  site  ############################################
###############################################################################################
def site((Lx,Ly),x,y):
  return x + (y*Lx)  #convert (x,y) pair to a single site number
