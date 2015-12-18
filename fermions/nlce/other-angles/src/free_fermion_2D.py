import numpy as np
import pprint

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

    Eval,Evec = np.linalg.eigh(Hamiltonian) 
    U=np.matrix(Evec)
    #print Eval

    '''
    GSOccup=np.kron(([1,0],[0,0]),np.eye(Lx*Ly))
    C= U*GSOccup*U.getH()
    '''

    GSU=U[:,0:Lx*Ly]
    C=GSU*GSU.getH()


    Cred=C[np.ix_(sitesA,sitesA)]

    Ev = abs(np.linalg.eigvals(Cred).real)
    #print Ev

    Sn = np.zeros(len(alpha))
    #Sn = np.ones(len(alpha))
    
    vonNeumannS=0.0
    vonNeumannS = - sum(Ev*np.log(Ev+1e-15)) - sum((1.-Ev)*np.log(abs(1.-Ev+1e-15)))
    for i,n in enumerate(alpha):
        if n==1:
            Sn[i]=vonNeumannS
        else:
            Sn[i]=(1./(1.-n))*sum(np.log((Ev+1e-15)**n + (abs(1-Ev+1e-15))**n))
            #Sn[i]=(1./(1.-n))*(Sn[i])
    #print Sn
    return Sn
#........................................END getEntropy........................................

###############################################################################################
########################################  getRegionA  #########################################
###############################################################################################

def getRegionA((Lx,Ly),(x0,y0),bipart,angle=90): #all factors of two due to fermion spin 
    regA = np.zeros( (2*Lx,Ly), dtype='bool' ) #doubling in x direction only!


    if angle==90:
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            regA[:,:y0]=True
        elif bipart==3:
            regA[2*x0:,y0:]=True
        else:
            regA[:2*x0,:y0]=True

    if angle==22.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(2*x0-3,-1,-2)):
                regA[e,0:min(Ly,y0+2*i+2)]=True
                regA[e-1,0:min(Ly,y0+2*i+2)]=True

            regA[2*x0-1,0:min(Ly,y0)]=True
            regA[2*x0-2,0:min(Ly,y0)]=True
            for i,e in enumerate(range(2*x0,2*Lx,2)):
                regA[e,0:max(0,y0-2*i)]=True
                regA[e+1,0:max(0,y0-2*i)]=True
        elif bipart==3:
            for i,e in enumerate(range(2*x0-1,-1,-2)):
                regA[e,min(y0+2*i,Ly):]=True
                regA[e-1,min(y0+2*i,Ly):]=True


    if angle==45:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(2*x0-1,-1,-2)):
                regA[e,0:min(Ly,y0+i+1)]=True
                regA[e-1,0:min(Ly,y0+i+1)]=True

            for i,e in enumerate(range(2*x0,2*Lx,2)):
                regA[e,0:max(0,y0-i)]=True
                regA[e+1,0:max(0,y0-i)]=True
            #concave corners of the pixelized line are the locations of the (x0,y0)
        elif bipart==3:
            for i,e in enumerate(range(2*x0-1,-1,-2)):
                regA[e,min(y0+i,Ly):]=True
                regA[e-1,min(y0+i,Ly):]=True

    if angle==67.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(2*x0-3,-1,-4)):
                regA[e,0:min(Ly,y0+i+1)]=True
                regA[e-1,0:min(Ly,y0+i+1)]=True
                if e>2:
                    regA[e-2,0:min(Ly,y0+i+1)]=True
                    regA[e-3,0:min(Ly,y0+i+1)]=True

            regA[2*x0+1,0:min(Ly,y0)]=True
            regA[2*x0,0:min(Ly,y0)]=True
            regA[2*x0-1,0:min(Ly,y0)]=True
            regA[2*x0-2,0:min(Ly,y0)]=True
            for i,e in enumerate(range(2*x0+2,2*Lx,4)):
                regA[e,0:max(0,y0-i-1)]=True
                regA[e+1,0:max(0,y0-i-1)]=True
                if e<2*Lx-3:
                    regA[e+2,0:max(0,y0-i-1)]=True
                    regA[e+3,0:max(0,y0-i-1)]=True
        elif bipart==3:
            regA[2*x0-1,y0:Ly]=True
            regA[2*x0-2,y0:Ly]=True
            for i,e in enumerate(range(2*x0-3,-1,-4)):
                regA[e,min(y0+i+1,Ly):]=True
                regA[e-1,min(y0+i+1,Ly):]=True
                if e>2:
                    regA[e-2,min(y0+i+1,Ly):]=True
                    regA[e-3,min(y0+i+1,Ly):]=True


    if angle==112.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(2*x0-3,-1,-4)):
                regA[e,0:min(Ly,y0+i+1)]=True
                regA[e-1,0:min(Ly,y0+i+1)]=True
                if e>2:
                    regA[e-2,0:min(Ly,y0+i+1)]=True
                    regA[e-3,0:min(Ly,y0+i+1)]=True

            regA[2*x0+1,0:min(Ly,y0)]=True
            regA[2*x0,0:min(Ly,y0)]=True
            regA[2*x0-1,0:min(Ly,y0)]=True
            regA[2*x0-2,0:min(Ly,y0)]=True
            for i,e in enumerate(range(2*x0+2,2*Lx,4)):
                regA[e,0:max(0,y0-i-1)]=True
                regA[e+1,0:max(0,y0-i-1)]=True
                if e<2*Lx-3:
                    regA[e+2,0:max(0,y0-i-1)]=True
                    regA[e+3,0:max(0,y0-i-1)]=True
        elif bipart==3:
            regA[2*x0-1,y0:Ly]=True
            regA[2*x0-2,y0:Ly]=True
            for i,e in enumerate(range(2*x0-3,-1,-4)):
                regA[e,max(y0-i-1,0):]=True
                regA[e-1,max(y0-i-1,0):]=True
                if e>2:
                    regA[e-2,max(y0-i-1,0):]=True
                    regA[e-3,max(y0-i-1,0):]=True



    if angle==135:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            #concave corners of the pixelized line are the locations of the (x0,y0)
            triangle_length=y0+x0-1
            for i,e in enumerate(range(0,2*min(Lx,triangle_length),2)):
                regA[e,0:min(Ly,triangle_length-i)]=True
                regA[e+1,0:min(Ly,triangle_length-i)]=True
        elif bipart==3:
            for i,e in enumerate(range(2*x0-1,0,-2)):
                regA[e,max(0,y0-i):Ly]=True
                regA[e-1,max(0,y0-i):Ly]=True

    if angle==157.5:
        #y0 must be at least 1
        if bipart==1:
            regA[:2*x0,:]=True
        elif bipart==2:
            for i,e in enumerate(range(2*x0-3,-1,-2)):
                regA[e,0:min(Ly,y0+2*i+2)]=True
                regA[e-1,0:min(Ly,y0+2*i+2)]=True

            regA[2*x0-1,0:min(Ly,y0)]=True
            regA[2*x0-2,0:min(Ly,y0)]=True
            for i,e in enumerate(range(2*x0,2*Lx,2)):
                regA[e,0:max(0,y0-2*i)]=True
                regA[e+1,0:max(0,y0-2*i)]=True
                
        elif bipart==3:
            #regA[x0-1,y0:Ly]=True
            for i,e in enumerate(range(2*x0-1,-1,-2)):
                regA[e,max(y0-2*i,0):]=True
                regA[e-1,max(y0-2*i,0):]=True

    #print regA.T

    if( (regA==True).sum() > (2*Lx*Ly/2) ): 
        regA = np.logical_not(regA)
    

    return regA

###############################################################################################
###########################################  site  ############################################
###############################################################################################
def site((Lx,Ly),x,y):
  return x + (y*Lx)  #convert (x,y) pair to a single site number
