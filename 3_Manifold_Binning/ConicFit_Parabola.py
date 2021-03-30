import numpy as np
import sympy

# =============================================================================
# Least squares fit each 2D subspace per CM with constrained conic (parabola)
# =============================================================================
# Codified:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================

def fit1(U_rot_Nd1, v1, v2): #restrained conic fit (parabola)
    # =============================================================================
    # Paper: 'Direct and Specific Fitting of Conics to Scattered Data'
    # Authors: Harker, O'Leary, et al.
    # =============================================================================
    # Convert to 'mean-free' coordinates:
    mu1 = np.mean(U_rot_Nd1[:,v1])
    mu2 = np.mean(U_rot_Nd1[:,v2])
    U_rot_Nd0 = np.copy(U_rot_Nd1)
    for i in range(np.shape(U_rot_Nd0)[0]):
        U_rot_Nd0[i,v1] = U_rot_Nd1[i,v1] - np.mean(U_rot_Nd1[:,v1])
        U_rot_Nd0[i,v2] = U_rot_Nd1[i,v2] - np.mean(U_rot_Nd1[:,v2])   
    if 0: #confirm that data is mean free
        print(np.mean(U_rot_Nd0[:,v1]))
        print(np.mean(U_rot_Nd0[:,v2]))
    m = np.sqrt(2.*np.shape(U_rot_Nd0)[0] / (sum(U_rot_Nd0[:,v1]**2 + U_rot_Nd0[:,v2]**2)))
    U_rot_Nd0[:,v1] = m*U_rot_Nd0[:,v1]
    U_rot_Nd0[:,v2] = m*U_rot_Nd0[:,v2]
    if 0: #confirm that data is scaled to have root-mean-square distance sqrt(2) from origin
        rmsd = 0
        for i in range(np.shape(U_rot_Nd0)[0]):
            rmsd += np.sqrt(U_rot_Nd0[i,v1]**2 + U_rot_Nd0[i,v2]**2)**2 / np.shape(U_rot_Nd0)[0]
        print(np.sqrt(rmsd))
    
    # General equation for nondegenerate conic section: x^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    d_XX = U_rot_Nd0[:,v1]*U_rot_Nd0[:,v1] #x^2
    d_XY = U_rot_Nd0[:,v1]*U_rot_Nd0[:,v2] #xy
    d_YY = U_rot_Nd0[:,v2]*U_rot_Nd0[:,v2] #y^2
    d_X = U_rot_Nd0[:,v1] #x
    d_Y = U_rot_Nd0[:,v2] #y
    # Design matrices:
    D2 = np.column_stack((d_XX - np.mean(d_XX), d_XY - np.mean(d_XY), d_YY - np.mean(d_YY))) #quadratic form
    D1 = np.column_stack((d_X, d_Y)) #linear form
    D0 = np.ones(np.shape(d_XX)[0]) #constant terms
    D = np.column_stack((D2, D1, D0))
    # Scatter matrices:
    S22 = np.matmul(D2.T, D2)
    S21 = np.matmul(D2.T, D1)
    S20 = np.matmul(D2.T, D0)
    S12 = np.matmul(D1.T, D2)
    S11 = np.matmul(D1.T, D1)
    S10 = np.matmul(D1.T, D0)
    S02 = np.matmul(D0.T, D2)
    S01 = np.matmul(D0.T, D1)
    S00 = np.matmul(D0.T, D0)
    S11i = np.linalg.inv(S11)
    #print((-1./np.shape(U_rot_Nd0)[0])*(np.array([S20.T,S10.T]))) #should be 0-vector
    M = S22 - (S21).dot(S11i).dot(S21.T)
    if 0: #ellipse/hyperbola
        C = np.zeros(shape=(3,3)) #constraint matrix
        C[0,2] = -2.
        C[2,0] = -2.
        C[1,1] = 1.
        Ci = np.linalg.inv(C)
        eVals, eVecs = np.linalg.eig(np.matmul(Ci,M))
        eIdx = eVals.argsort()[::-1]
        eVals = eVals[eIdx]
        eVecs = eVecs[:,eIdx]
        if 1: #ellipse
            z2 = eVecs[:,-1]
        else: #hyperbola
            z2 = eVecs[:,-2]
    else: #parabola
        C = np.eye(3)
        Ci = np.linalg.inv(C)
        eVals, eVecs = np.linalg.eig(np.matmul(Ci,M))
        eIdx = np.abs(eVals).argsort()[::-1] 
        eVals = eVals[eIdx]
        eVecs = eVecs[:,eIdx]
        # Note: z2 = eVecs[:,2]; smallest singular value is best approximation (so far) for conic...
        # Further optimization requires linear combination of eigenvectors:
        e11 = eVecs[0,0]
        e12 = eVecs[1,0]
        e13 = eVecs[2,0]
        e21 = eVecs[0,1]
        e22 = eVecs[1,1]
        e23 = eVecs[2,1]
        e31 = eVecs[0,2]
        e32 = eVecs[1,2]
        e33 = eVecs[2,2]
        g1 = e22**2 - 4.*e21*e23 
        g2 = 2.*e22*e12 - 4.*e21*e13 - 4.*e11*e23
        g3 = e12**2 - 4.*e11*e13
        g4 = 2.*e32*e22 - 4.*e31*e23 - 4.*e21*e33
        g5 = 2*e32*e12 - 4*e31*e13 - 4*e11*e33
        g6 = e32**2 - 4.*e31*e33
        # Define coefficients of fourth order polynomial:
        alpha1 = eVals[0]**2
        alpha2 = eVals[1]**2
        alpha3 = alpha1*alpha2
        k1 = 4.*g3*g6 - g5**2
        k2 = g2*g6 - (1/2.)*g4*g5
        k3 = (1/2.)*g2*g5 - g3*g4
        k4 = 4.*g6*g1 - g4**2
        k5 = 4.*g1*g3 - g2**2
        k6 = g2*g4 - 2.*g1*g5
        k7 = -4.*(g1*alpha1 + alpha2*g3)
        k8 = g1*k1 - g2*k2 + g4*k3
        # Polynomial coefficients:
        K0 = 16.*g6*alpha3**2
        K1 = -8.*alpha3*(k1*alpha2 + k4*alpha1)
        K2 = 4.*((2.*g2*k2 + 4.*k8)*alpha3 + g1*k4*alpha1**2 + g3*k1*alpha2**2)
        K3 = 2.*k7*k8
        K4 = k5*k8
        polyCoeffs = [K4, K3, K2, K1, K0]
        polyRoots = np.real(np.roots(polyCoeffs))
        polyIdx = np.abs(polyRoots).argsort()[::-1] 
        polyRoots = polyRoots[polyIdx]
        mu_star = polyRoots[-1] #best fit corresponds to real Lagrange multipiler with smallest magnitude
        u_star = k5*mu_star**2 + k7*mu_star + 4*alpha3
        s_star = ((2.*mu_star)/u_star)*(k3*mu_star + alpha1*g4)
        t_star = (mu_star/u_star)*(k6*mu_star + 2.*alpha2*g5)
        z2 = eVecs[:,2] + s_star*eVecs[:,1] + t_star*eVecs[:,0]

    # Backsubstitution:
    B = np.zeros(shape=(6,3))
    B[0:3,0:3] = np.eye(3)
    B[3:5,:] = -1.*np.matmul(np.linalg.inv(S11), S21.T)
    B[5,0] = -1.*np.mean(d_XX)
    B[5,1] = -1.*np.mean(d_XY)
    B[5,2] = -1.*np.mean(d_YY)
    cF = np.matmul(B, z2)
    
    K = np.array([[cF[0], cF[1]/2., cF[3]/2.], [cF[1]/2., cF[2], cF[4]/2.], [cF[3]/2., cF[4]/2., cF[5]]])
    T = np.array([[m, 0, -1.*m*mu1],[0, m, -1.*m*mu2],[0, 0, 1]])
    K_star = (T.T).dot(K).dot(T)
    cF[0] = K_star[0,0]
    cF[1] = K_star[0,1]*2.
    cF[2] = K_star[1,1]
    cF[3] = K_star[0,2]*2.
    cF[4] = K_star[1,2]*2.
    cF[5] = K_star[2,2]
        
    # Fitting errors:
    r_Dz = np.matmul(D, cF)
    ssr = np.matmul(r_Dz.T, r_Dz) #sum of squared residuals
    
    # =========================================================================
    # Find angle to align parabola with Cartesian plane:
    # Textbook: Calculus - Section 'Rotation of Axes'
    # Author: James Stewart
    # =========================================================================
    adj = cF[0]-cF[2] #A-C
    opp = cF[1] #B
    hyp = np.sqrt(adj**2 + opp**2)
    if cF[1] < 0:
        cos2Th = np.abs(adj/hyp)
        cosTh = np.sqrt((1+cos2Th)/2.)
        sinTh = np.sqrt((1-cos2Th)/2.)
        Theta = np.arccos(cosTh)
    else:            
        cos2Th = np.abs(adj/hyp)
        cosTh = np.sqrt((1+cos2Th)/2.)
        sinTh = -1.*np.sqrt((1-cos2Th)/2.)
        Theta = -1.*np.arccos(cosTh)

    R = np.array([[np.cos(Theta), np.sin(Theta)],[-1.*np.sin(Theta), np.cos(Theta)]])
    U_rot_2d = np.matmul(R, U_rot_Nd1[:,[v1,v2]].T)
    U_rot_2d = U_rot_2d.T                
    # =========================================================================
    # Transform coefficients:
    # Website: Wikipedia: 'Rotation of Conic Sections'
    # =========================================================================
    cF0 = np.zeros(6)
    cF0[0] = (cF[0]*np.cos(Theta)**2) + (cF[1]*np.sin(Theta)*np.cos(Theta)) + (cF[2]*np.sin(Theta)**2)
    cF0[1] = 2.*(cF[2]-cF[0])*np.sin(Theta)*np.cos(Theta) + cF[1]*(np.cos(Theta)**2 - np.sin(Theta)**2)
    cF0[2] = (cF[0]*np.sin(Theta)**2) - (cF[1]*np.sin(Theta)*np.cos(Theta)) + (cF[2]*np.cos(Theta)**2)
    cF0[3] = cF[3]*np.cos(Theta) + cF[4]*np.sin(Theta)
    cF0[4] = -1.*cF[3]*np.sin(Theta) + cF[4]*np.cos(Theta)
    cF0[5] = cF[5]
    # Simplify xy and y^2 term to zero (if within threshold of zero):
    if -1e10 < cF0[1] < 1e10:
        cF0[1] = 0.
    else:
        print('Nonzero found: B')
    if -1e10 < cF0[1] < 1e10:
        cF0[2] = 0.
    else:
        print('Nonzero found: C')  
    # Calculate coefficient of determination:
    def ParabolicFit(x, A, D, E, F):
        return (A*x**2 + D*x + F)/(-1.*E) #d + b*(a*x + c)**2
    Pcoff = [cF0[0],cF0[3],cF0[4],cF0[5]]
    ss_res = 0
    for i in range(len(U_rot_2d[:,0])):
        ss_res += (U_rot_2d[i,1] - ParabolicFit(U_rot_2d[i,0], *Pcoff))**2 #residual sum of squares
    ybar = np.mean(U_rot_2d[:,1])
    ss_tot = 0
    for i in range(len(U_rot_2d[:,0])):
        ss_tot += (U_rot_2d[i,1] - ybar)**2 #total sum of squares            
    R2 = 1. - (ss_res / ss_tot) #coefficient of determination
    return np.array([cF, cF0, Theta, R2], dtype='object')


def fit2(U_rot_Nd1, v1, v2): #most general conic fit (open to any conic)
    # General equation for nondegenerate conic section: 
    # x^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    A0 = U_rot_Nd1[:,v1]*U_rot_Nd1[:,v1] #x^2
    B = U_rot_Nd1[:,v1]*U_rot_Nd1[:,v2] #xy
    C = U_rot_Nd1[:,v2]*U_rot_Nd1[:,v2] #y^2
    D = U_rot_Nd1[:,v1] #x
    E = U_rot_Nd1[:,v2] #y
    F = np.ones(np.shape(A0)[0]) #constants
    A = np.column_stack((B,C,D,E,F))
    b = -1.*A0 #move over constants
    AtA = np.matmul(A.T, A)
    Atb = np.matmul(A.T, b)

    if 1: #solve system via Gaussian elimination
        A_aug = np.column_stack((AtA, Atb))
        rref = sympy.Matrix(A_aug).rref(simplify=True)
        rref_arr = np.array(rref[0].tolist(), dtype=float)
        cF = rref_arr[:,-1] #coefficients
    else: #solve system via LU factorization
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(AtA)
        cF = lu_solve((lu, piv), Atb) #coefficients B,C,D,E,F
        
    cF1 = np.array([1., cF[0], cF[1], cF[2], cF[3], cF[4]]) #A,B,C,D,E,F

    # Fitting errors:
    tss = np.sum((b-np.mean(b))**2) #total sum of squares
    ssr = np.sum((b-np.matmul(A,cF))**2) #sum of square residuals
    R2 = 1-(ssr/tss) #coefficient of determination
    
    # =========================================================================
    # Find angle to align parabola with Cartesian plane:
    # Textbook: Calculus - Section 'Rotation of Axes'
    # Author: James Stewart
    # =========================================================================
    adj = 1.-cF[1] #A-C
    hyp = np.sqrt((adj)**2 + cF[0]**2)
    cos2Th = adj/hyp
    cosTh = np.sqrt((1+cos2Th)/2.) #half-angle identity
    if cF1[1] > 0: #due to demoninator of (y/x), angle must be reversed (i.e., atan2)
        sinTh = -1.*np.sqrt((1-cos2Th)/2.) #half-angle identity
        Theta = np.arccos(cosTh) #theta [radians]
    else:
        sinTh = np.sqrt((1-cos2Th)/2.) #half-angle identity
        Theta = -1.*np.arccos(cosTh) #theta [radians]
        
    # =========================================================================
    # Transform coefficients:
    # Website: Wikipedia: 'Rotation of Conic Sections'
    # =========================================================================
    cF0 = np.zeros(6)
    cF0[0] = (cF1[0]*np.cos(Theta)**2) + (cF1[1]*np.sin(Theta)*np.cos(Theta)) + (cF1[2]*np.sin(Theta)**2)
    cF0[1] = 2.*(cF1[2]-cF1[0])*np.sin(Theta)*np.cos(Theta) + cF1[1]*(np.cos(Theta)**2 - np.sin(Theta)**2)
    cF0[2] = (cF1[0]*np.sin(Theta)**2) - (cF1[1]*np.sin(Theta)*np.cos(Theta)) + (cF1[2]*np.cos(Theta)**2)
    cF0[3] = cF1[3]*np.cos(Theta) + cF1[4]*np.sin(Theta)
    cF0[4] = -1.*cF1[3]*np.sin(Theta) + cF1[4]*np.cos(Theta)
    cF0[5] = cF1[5]

    return np.array([cF1, cF0, Theta, R2], dtype='object')


def fit3(U_rot_Nd, v1, v2): #simple parabolic fit (no 'xy' cross terms)
    A1 = U_rot_Nd[:,v1]**2
    A2 = U_rot_Nd[:,v1]
    A3 = np.ones(np.shape(A1)[0])
    A = np.column_stack((A1,A2,A3))
    b = U_rot_Nd[:,v2]
    AtA = np.matmul(A.T, A)
    Atb = np.matmul(A.T, b)
    A_aug = np.column_stack((AtA, Atb))
    rref = sympy.Matrix(A_aug).rref()
    rref_arr = np.array(rref[0].tolist(), dtype=float)
    cF = rref_arr[:,-1]
    # Fitting errors:
    tss = np.sum((b-np.mean(b))**2) #total sum of squares
    ssr = np.sum((b-np.matmul(A,cF))**2) #sum of square residuals
    R2 = 1-(ssr/tss) #coefficient of determination
    return np.array([cF, R2], dtype='object')

