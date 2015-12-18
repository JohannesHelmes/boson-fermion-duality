class Max:

  def clusters(self,N):
    result=[]
    y=N
    for x in range(2,N): #in 1xN clusters is no corner
        result.append((x,y))
        result.append((y,x)) #NON ISOTROPIC VARIANT
    result.append((N,N))
    return result
  
  def clusters_fix_sum(self,N,s):
    result=[]
    minX=max(2,s-N)
    maxX=min(N,s/2)
    

    for x in range(minX,maxX+1): #in 1xN clusters is no corner
        result.append((x,s-x))
        if x!=(s-x): # NON ISOTROPIC VARIANT
            result.append((s-x,x))
    return result

  def name(self):
    return "max"

#NOT DOWNGRADED TO 2D !!!!!!!!!!!!!!!!!!!!!!!!!!!!
# class Arithmetic:
# 
#     def length(self,N):
#         return (1+N)
#     
#     def lengthstr(self,N):
#         return "%.1f"%self.length(N)
#     
#     def clusters(self,N):
#         x = N-1
#         y = 1
#         res = [(x,y),]
#         while x > (y+1):
#             x -= 1
#             y += 1
#             res.append((x,y))
#         return res
# 
# class Geometric:
# 
#     def clusters(self,N):
#         res = [(N,1),]
#         i = 2
#         sqrtN = N ** (0.5)
#         while i < sqrtN:
#             if N % i == 0:
#                 res.append((N/i,i))
#             i += 1
#         if i*i == N:
#             res.append((int(sqrtN),int(sqrtN)))
#         
#         return res
        
