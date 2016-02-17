import clust_order
import numpy as np
import os.path  #to check if file exists
import sys  #for sys.stdout.flush()
import time
import free_boson_2D
from mpi4py import MPI

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

alphaEqTol=1e-3
path="../data/"

def arrayEq(a1, a2):
  eq = True
  
  if len(a1) != len(a2):
    print "length different"
    eq = False
  else:
    for elem1, elem2 in zip(a1, a2):
      #if elem1 != elem2:
      #Check if the alpha values are equal (within tolerance):
      if abs(elem1-elem2) > alphaEqTol:
        eq = False
        print "%.20f neq %.20f" %(elem1,elem2)
  #end if-else
  
  return eq
####### end arrayEq(a1, a2) function #######

def decimalStr(num):
  res = str(num)
  length = len(res)
  index = res.find('.')
  if index >= 0:
    res = res[0:index] + '-' + res[(index+1):length]
  return res
####### end decimalStr(num) function #######

def readArray(line):
  start = max( 0, line.find('[') )
  end = min( len(line), line.find(']') )
  return line[start+1:end] 
####### end readArray(line) function #######

def readWeights(alpha, massterm, max_order):
  w={}
  filename = path+"weights_mass" + decimalStr(massterm) + ".txt"
  if os.path.isfile(filename):
    fin = open(filename,'r')
    line = fin.readline()
    alpha_read = readArray(line).split()
    alpha_read = [float(a) for a in alpha_read]
    
    #if the alpha list is the same, then can use these previous stored weights:
    if arrayEq(alpha, alpha_read):
      line = fin.readline() #read blank line
      
      line = fin.readline()
      while line != "":
        #Read the name of the cluster:
        clust_name = line[ 0 : min(line.find(':'),len(line)) ]
        
        #Read the weights of that cluster for each alpha:
        weightArr = readArray(line).split()
        weightArr = np.array([float(elem) for elem in weightArr])
        
        #Store these weights:
        w[clust_name] = weightArr
        
        line = fin.readline()
      #end loop over lines
    else:
      print "WARNING: alpha list for the weights stored in " + filename + " does not match " + \
      "the current alpha list"
    
    fin.close()

    weights=np.zeros((max_order+1,max_order+1,len(alpha)))

    for name,entry in w.items():
        lx=int(name.split('x')[0])
        ly=int(name.split('x')[1])
        if (lx<=max_order)and(ly<=max_order):
            weights[lx,ly]=entry
  else:
    weights=np.zeros((max_order+1,max_order+1,len(alpha)))


  return w,weights

####### end readWeights(alpha, massterm) #######

def clust_name(Lx,Ly):
    return "%02dx%02dx"%(Lx,Ly)


def getXandP(Lx,Ly,massterm):
    Ns = Lx * Ly
    L  = (Lx,Ly)

    K=(4.0 + massterm**2)*np.eye(Ns)+np.kron(np.eye(Ly),np.diag([-1.0]*(Lx-1),1)+np.diag([-1.0]*(Lx-1),-1))
    if Lx>0:
        K+=np.diag([-1.]*(Ns-Lx),Lx)+np.diag([-1.]*(Ns-Lx),-Lx)

    Eval,Evec = np.linalg.eigh(K) #, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False, check_finite=True)
    EvMat=np.matrix(Evec)
    EvaSqrt=np.sqrt(Eval)

    P = 1./2. * EvMat * np.diag(EvaSqrt) * EvMat.T
    X = 1./2. * EvMat * np.diag(1. /EvaSqrt) * EvMat.T
    

    return X,P

#############################
# User settings

order_min = 2
order_max = 55 
order = clust_order.Max()
massterm = 0.0
#############################


clusters = []
alpha= 1./np.linspace(0.2,4,39)
alpha= (1./np.linspace(1.1,5,40)).tolist() + np.linspace(1.0,5.0,41).tolist()

if rank==0:
    t1 = time.clock()
    total = np.zeros(len(alpha))
    w,weights = readWeights(alpha,massterm,order_max) #try to read in weights (if there are any stored)
    print "\nInitial weights:"
    print [key for key in w]
    #print w

    #Save the weights to file:
    filename = path+"weights_mass" + decimalStr(massterm) + ".txt"
    if os.path.isfile(filename):
        fout_w = open(filename, 'a')
    else:
        fout_w = open(filename, 'w')
        #Write the alpha array:
        fout_w.write("alpha = [ ")
        for n in alpha:
          fout_w.write("%.3f\t" %n)
        fout_w.write(" ]\n\n")


    fout_res=[0 for i in alpha]
    for i,n in enumerate(alpha):
      filename = path+"results_mass" + decimalStr(massterm) + "_alpha" + decimalStr(n) + ".txt"
      fout_res[i] = open(filename, 'w')
    #keylist=w.keys()
else:
    weights=np.zeros((order_max+1,order_max+1,len(alpha)))
    #keylist=[]
  

MaxSUM=2*order_max

#print rank, keylist
#comm.Bcast(keylist)
for equalsum in range(2*order_min,MaxSUM+1):
    comm.Bcast(weights)

    for i,(Lx,Ly) in enumerate(order.clusters_fix_sum(order_max,equalsum)):
        curr_clust_name = clust_name(Lx,Ly)
        sys.stdout.flush()


        if weights[Lx,Ly].all()==0:
            X,P=getXandP(Lx,Ly,massterm)
            
            w_clust_name = clust_name(Lx,Ly)

            if rank==0:
                print Lx,"x",Ly
                print "    Must call free boson solver"

            horEntros= np.zeros((Ly/2+1,len(alpha)))
            verEntros= np.zeros((Lx,len(alpha)))
            cornerEntros_ll= np.zeros((Lx,Ly/2+1,len(alpha)))
            cornerEntros_ur= np.zeros((Lx,Ly/2+1,len(alpha)))
            #First term in weight of this cluster is the property of the cluster:
            #w[w_clust_name] = free_boson_2D.getCornerEnt(Lx,Ly,alpha,massterm)

            local_horEntros= np.zeros((Ly/2+1,len(alpha)))
            local_verEntros= np.zeros((Lx,len(alpha)))
            local_cornerEntros_ll= np.zeros((Lx,Ly/2+1,len(alpha)))
            local_cornerEntros_ur= np.zeros((Lx,Ly/2+1,len(alpha)))

            L=(Lx,Ly)

            #DO THE STUFF
            i=0
            for y0 in range(1,Ly/2+1):
                if rank==i%size:
                    local_horEntros[y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,(0,y0),2),X,P)
                i+=1
            
            for x0 in range(1,Lx):
                if rank==i%size:
                    local_verEntros[x0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,(x0,0),1),X,P)
                i+=1

            for y0 in range(1,(Ly+1)/2):
                for x0 in range(1,Lx):
                    r0=(x0,y0)
                    if rank==i%size:
                        local_cornerEntros_ll[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,4),X,P)
                    i+=1
                    if rank==i%size:
                        local_cornerEntros_ur[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,3),X,P)
                    i+=1
            
            if Ly%2==0:
                y0=Ly/2
                maxX=(Lx+1)/2
                for x0 in range(1,maxX):
                    r0 = (x0,y0) #position of the corner
                    #corners:
                    if rank==i%size:
                        local_cornerEntros_ll[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,4),X,P)
                    i+=1
                    if rank==i%size:
                        local_cornerEntros_ur[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,3),X,P)
                    i+=1

                x0=Lx/2
                y0=Ly/2
                if (Lx%2==0)and(Ly%2==0):
                    r0 = (x0,y0) #position of the corner
                    #corners:
                    if rank==i%size:
                        local_cornerEntros_ll[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,4),X,P)
                    i+=1
                    if rank==i%size:
                        local_cornerEntros_ur[x0,y0]= free_boson_2D.getEntropy(L,alpha,free_boson_2D.getRegionA(L,r0,3),X,P)
                    i+=1


            comm.Reduce(local_horEntros,horEntros,op=MPI.SUM)
            comm.Reduce(local_verEntros,verEntros,op=MPI.SUM)
            comm.Reduce(local_cornerEntros_ur,cornerEntros_ur,op=MPI.SUM)
            comm.Reduce(local_cornerEntros_ll,cornerEntros_ll,op=MPI.SUM)


            if rank==0:
                for y0 in range(1,(Ly+1)/2):
                    for x0 in range(1,Lx):
                        r0 = (x0,y0) #position of the corner
                        #corners:
                        weights[Lx,Ly]+= cornerEntros_ll[x0,y0] + cornerEntros_ur[x0,y0] - horEntros[y0] - verEntros[x0]
                if Ly%2==0:
                    y0=Ly/2
                    maxX=(Lx+1)/2
                    for x0 in range(1,maxX):
                        r0 = (x0,y0) #position of the corner
                        #corners:
                        weights[Lx,Ly]+= cornerEntros_ll[x0,y0] + cornerEntros_ur[x0,y0] - horEntros[y0] - verEntros[x0]


                    x0=Lx/2
                    y0=Ly/2
                    if (Lx%2==0)and(Ly%2==0):
                        r0 = (x0,y0) #position of the corner
                        #corners:
                        weights[Lx,Ly]+= 0.5 * (cornerEntros_ll[x0,y0] + cornerEntros_ur[x0,y0] - horEntros[y0] - verEntros[x0] )


                # Subtract weights of all subclusters:
                for x in range(2, Lx+1):
                    for y in range(2, Ly+1):
                      # Check that we are not at the Lx x Ly x Lz cluster, and also check that we are not at
                      # the 1 x 1 x 1 cluster (both of these cases don't contribute:
                        if (x!=Lx or y!=Ly) and max(x,y)>1:
                        
                            coeff = (Lx-x+1)*(Ly-y+1)
                            
                            #weight is stored such that x<=y<=z, so sort the current x,y,z ONLY in isotropic case
                            [xs,ys] = [x,y] if x<=y else [y,x] #NON ISOTROPIC VARIANT
                            
                            #w[w_clust_name] -= coeff * w[clust_name(xs,ys)]
                            weights[Lx,Ly] -= coeff * weights[xs,ys]

        else:
            if rank==0:
                print "    Calculated previously"

        
    if rank==0:
        print "Done Equalsum", equalsum
        print
        for Lx,Ly in order.clusters_fix_sum(order_max,equalsum):
            #Write the new weight to file:
            curr_clust_name = clust_name(Lx,Ly)
            w[curr_clust_name]=weights[Lx,Ly]
            fout_w.write(curr_clust_name + ": [ ")
            for i in range(len(alpha)):
              fout_w.write("%.20e\t" %weights[Lx,Ly][i])
            fout_w.write(" ]\n")
            fout_w.flush()
    #end loop over loop over clusters
        
if rank==0:

    for ord in range(order_min,order_max+1):
        for Lx,Ly in order.clusters(ord):
            ef = 1 if Lx==Ly else 2 # TESTING NON ISOTROPIC CASE
            total = total + ef * weights[Lx,Ly] 


        # Save result to file
        for i in range(len(alpha)):
            fout_res[i].write("%d %.15f"%(ord,total[i])+'\n')
            fout_res[i].flush()



    for fout in fout_res:
      fout.close()

    fout_w.close()

    print "\nFinal weights:"
    print [key for key in w]

    print "\nOrder done: ",str(order_max)
    print

    t2 = time.clock()
    print "Total time: " + str(t2-t1) + " sec."
