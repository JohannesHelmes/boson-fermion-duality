#!/usr/bin/env python
# encoding: utf-8
"""
MutInfoDiracFermions.py

Created by Anushya Chandran on 2015-06-30.

"""


import EntanglementFuncs_Haldane as efunc
import sys
import argparse
import os
import numpy as np
import time

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Determining 2f for a linearly dispersing Dirac fermion on a square lattice from the mutual information of concentric circles ')
    
    parser.add_argument('--M', type=float, help='Mass parameter of the Haldane model', default = 0.0)
    parser.add_argument('--bc', type=str, help='Boundary conditions along the x-direction. Allowed values: pbc, apbc', default = 'pbc')
    
    parser.add_argument('--epsx','-x', type=float, help='x coordinate of the center of the circle minus system size/2', default=0.3)
    
    parser.add_argument('--epsy','-y', type=float, help='y coordinate of the center of the circle minus system size/2', default=0.3)
    
    parser.add_argument('--radius','-r',nargs=3, type=float, help='radius (min, max, step)')
    parser.add_argument('--width','-w', type=float, help='Width of the annulus. If set to zero, compute the entanglement entropy of the region.')
    parser.add_argument('--shape','-s', type=str, help='Shape of the region', default='circle')
    
    parser.add_argument('--Lmult','-m', type=float, help='Ratio of system size L to radius r', default=10)
    
    parser.add_argument('--outDir', type=str, help='Directory for output file', default='.')

    args = parser.parse_args()

    M = args.M
    bc = args.bc
    epsx = args.epsx
    epsy = args.epsy
    
    rmin = args.radius[0]
    rmax = args.radius[1]
    rstep = args.radius[2]
    width = args.width
    Lmult = args.Lmult
    shape = args.shape

    alphas=(1./np.linspace(1.1,5.0,40)).tolist()+np.linspace(1.0,5.0,41).tolist()
    alphas=[1.0,2.0,3.0,4.0]

    # Directory for data
    pathn = args.outDir+'/Data/'
    
    if not os.path.exists(pathn):
        os.makedirs(pathn)

    fnames=[]
    for alpha in alphas:
        fname = 'EE_'+shape+'_M%1.2f_Lmult%1.2f_alpha%1.3f_'%(M, Lmult, alpha)+bc+'.txt'
        fnames.append(pathn+fname)
        
        if not os.path.exists(pathn+fname):
            f = open( pathn+fname, 'w')
            
            #Write all the parameters of the run into the file as comments
            f.write('# M=%f\n'%M)
            f.write('# width=%f\n'%width)
            f.write('# Lmult=%f\n'%Lmult)
            f.write('# alpha=%f\n'%alpha)
            f.write('# epsx=%f\n'%epsx)
            f.write('# epsy=%f\n'%epsy)
            f.write('# bc='+bc+'\n')
            f.write('# shape='+shape+'\n')
            
            #Definitions of the columns of the array
            f.write('#Radius EE\n')
            f.close()
        
        
    rad_all = np.arange(rmin,rmax,rstep)

    #print fnames

    #print "Start at ",time.clock()
        
    efunc.ee_diffradii(shape, M, bc, epsx, epsy, Lmult, rad_all, alphas, fnames)
    
    return

if __name__ == '__main__':
  sys.exit(main())
