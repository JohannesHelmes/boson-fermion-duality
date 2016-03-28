import free_fermion_2D
import numpy as np
import argparse 

parser = argparse.ArgumentParser(description="Compute free fermion entanglement entropy for a single site using NLCE")
parser.add_argument('-b','--BC',choices=['pbc','yapbc','obc']) 

args = parser.parse_args()

BoundCond = args.BC


def makeRegionA_ss(Lx,Ly):
    regA = np.zeros( (2*Lx,Ly), dtype='bool' ) #doubling in x direction only!

    regA[Lx:Lx+2,Ly/2:Ly/2+1]=True

    return regA

def getHamiltonian(Lx,Ly,bc='pbc',massterm=1.0): #accepts also 'obc'
    Ns=Lx*Ly
    local_diag=np.diag([massterm,-massterm]) #True value on the diagonal is 2*m, but we achieve this by adding the hermitian conjugate.
    hopping_x_p=np.matrix(([-0.5*massterm, -0.5*1j],[-0.5*1j, 0.5*massterm]), dtype='complex')
    hopping_y_p=np.array(([-0.5*massterm, -0.5],[0.5, 0.5*massterm]))
    
    diag=np.kron(np.eye(Ns),local_diag)
    if bc=='pbc':
        diag_x=np.roll(np.eye(Lx),1,axis=1)
        hopping_x=np.kron(diag_x,hopping_x_p)
        offdiag_x=np.kron(np.eye(Ly),hopping_x)
    
        offdiag_y=np.kron(np.roll(np.eye(Ns),Lx,axis=1),hopping_y_p)
    
    if bc=='yapbc':
        diag_x=np.roll(np.eye(Lx),1,axis=1)
        hopping_x=np.kron(diag_x,hopping_x_p)
        offdiag_x=np.kron(np.eye(Ly),hopping_x)
    
        offdiag_y=np.kron(np.diag([1.]*(Ns-Lx),Lx),hopping_y_p)
        offdiag_y+=np.kron(np.diag([1.]*Lx,(Ns-Lx)),-hopping_y_p.T)
    
    elif bc=='obc':
        diag_x=np.diag([1.]*(Lx-1),1)
        hopping_x=np.kron(diag_x,hopping_x_p)
        offdiag_x=np.kron(np.eye(Ly),hopping_x)
        
        offdiag_y=np.kron(np.diag([1.]*(Ns-Lx),Lx),hopping_y_p)
    
    upper_tri=diag+offdiag_x+offdiag_y
    Ham=upper_tri+upper_tri.getH()
    
    
    return Ham

#print makeRegionA_ss(L,L)

for L in range(2,25,2):
    print L, free_fermion_2D.getEntropy((L,L),[1.0],makeRegionA_ss(L,L),getHamiltonian(L,L,BoundCond,1.0))[0]

