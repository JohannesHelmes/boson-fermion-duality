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
    parser.add_argument('--shape','-s', choices=['circle','square','triangle','parallelo45'], help='Shape of the region', default='square')

    parser.add_argument('--mode', choices=['ratio','singlesite'], help='Mode of bipartition', default='ratio')
    
    #mode ratio
    parser.add_argument('--radius','-r',nargs=3, type=float, help='radius (min, max, step)')
    parser.add_argument('--Lmult','-m', type=float, help='Ratio of system size L to radius r', default=10)
    
    #mode singlesite
    parser.add_argument('--L','-l',nargs=3, type=int, help='radius (min, max, step)')

    parser.add_argument('--outDir', type=str, help='Directory for output file', default='.')

    args = parser.parse_args()

    M = args.M
    bc = args.bc
    epsx = args.epsx
    epsy = args.epsy
    shape = args.shape
    
    
    if args.mode=='ratio':
        rmin = args.radius[0]
        rmax = args.radius[1]
        rstep = args.radius[2]
        rad_all = np.arange(rmin,rmax,rstep)
        L_all = np.array(rad_all*args.Lmult, dtype=int)
    elif args.mode=='singlesite':
        lmin = args.L[0]
        lmax = args.L[1]
        lstep = args.L[2]
        L_all = np.arange(lmin,lmax,lstep)
        rad_all = np.ones(len(L_all))
        

    alphas=(1./np.linspace(1.1,5.0,40)).tolist()+np.linspace(1.0,5.0,41).tolist()
    alphas=[1.0,2.0,3.0,4.0]
    alphas=[1.0]

    # Directory for data
    pathn = args.outDir
    
    if not os.path.exists(pathn):
        os.makedirs(pathn)

    fnames=[]
    for alpha in alphas:
        if args.mode=='ratio':
            fname = 'EE_'+shape+'_M%1.2f_Lmult%1.2f_alpha%1.3f_'%(M, args.Lmult, alpha)+bc+'.txt'
        elif args.mode=='singlesite':
            fname = 'EE_'+shape+'_M%1.2f_singlesite_alpha%1.3f_'%(M, alpha)+bc+'.txt'

        fnames.append(pathn+fname)
        
        if not os.path.exists(pathn+fname):
            f = open( pathn+fname, 'w')
            
            #Write all the parameters of the run into the file as comments
            f.write('# M=%f\n'%M)
            f.write('# alpha=%f\n'%alpha)
            f.write('# epsx=%f\n'%epsx)
            f.write('# epsy=%f\n'%epsy)
            f.write('# bc='+bc+'\n')
            f.write('# shape='+shape+'\n')
            
            #Definitions of the columns of the array
            f.write('#Radius EE\n')
            f.close()
        
        


    #print "Start at ",time.clock()

    print L_all, rad_all
        
    efunc.ee_diffradii(shape, M, bc, epsx, epsy, L_all, rad_all, alphas, fnames)
    
    return

if __name__ == '__main__':
  sys.exit(main())
