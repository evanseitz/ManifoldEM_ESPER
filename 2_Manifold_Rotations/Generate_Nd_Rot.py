import numpy as np
import math

def genNdRotations(n, theta): #args:{dimension, angle}    
    total = int(n*(n-1.)/2.) #n(n-1)/2 rotation matrices
    
    if isinstance(theta, list):
        if len(theta) != total:
            raise ValueError('%s angles required' % (total))
    if isinstance(theta, np.ndarray):
        if theta.shape[0] != total:
            raise ValueError('%sx1 array of angles required' % (total))
    
    R_all = np.zeros(shape=(n,n,total), dtype=float)

    i = 0
    j = 1
    k = 0
    for r in range(total):
        a11 = math.cos(theta[r])
        a12 = -1.*math.sin(theta[r])
        a21 = math.sin(theta[r])
        a22 = math.cos(theta[r])
        
        R_all[:,:,r] = np.eye(n)
        R_all[:,:,r][k,k] = a11
        R_all[:,:,r][k,j] = a12
        R_all[:,:,r][j,k] = a21
        R_all[:,:,r][j,j] = a22
        
        if j == (n-1):
            k += 1
            i = k
            j = k+1
        else:
            i += 1
            j += 1
            
        #print(R_all[:,:,r]) #individual rotation operator
        #print('')

    R_solo = R_all[:,:,0]
    for s in range(1,total):
        R_solo = R_solo.dot(R_all[:,:,s]) #form composite of all rotation operators
    return R_solo


# =============================================================================
# Example of use:
# =============================================================================

if 0:
    dim = 2
    theta_total = int(float(dim)*(float(dim)-1.)/2.)
    thetas = np.zeros(shape=(theta_total,1), dtype=float)
    for i in range(theta_total):
        thetas[i]=np.pi/2
    genNdRotations(dim, thetas) #5-dim; n(n-1)/2 = 10 angles