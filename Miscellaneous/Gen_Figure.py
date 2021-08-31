import sys, os, re
import numpy as np
import matplotlib
from matplotlib import rc
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pylab import imshow, show, axes
from mpl_toolkits.mplot3d import Axes3D

# =============================================================================
# Construction of parabolic surface from sinusoids (2 degrees of freedom)
# =============================================================================
# Run this script via 'python Gen_Figure.py' to generate Figure 5 in our...
# ...(pre-revised) bioRxiv manuscript. Variables have been kept consistent between our...
# ...detailed description therein and within this code.
# =============================================================================
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020-2021
# Contact:   evan.e.seitz@gmail.com
# =============================================================================


if 1: #Times font for all figures
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Cambria"] + plt.rcParams["font.serif"]

eta=25
k1=1
k2=2
k3=3
x = np.linspace(0,1,eta)
y1 = np.cos(k1*np.pi*x)
y2 = np.cos(k2*np.pi*x)
y3 = np.cos(k3*np.pi*x)
X0, Y0 = np.meshgrid(x, x) #only need X0

Y1 = np.ndarray(shape=(eta,eta))
for i in range(eta):
    Y1[i,:] = y1
Y2 = np.ndarray(shape=(eta,eta))
for i in range(eta):
    Y2[i,:] = y2

if 1: #cosine only (from higher-dimensional space)
    plt.scatter(X0, Y1, s=1)
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'$Y_1$')
    plt.show()

if 1: #Figure 5A; cosine grid (2D)
    plt.scatter(X0, Y1.T, s=1)
    plt.xlabel(r'$X_0$')
    plt.ylabel(r'${Y_1}^T$')
    plt.show()
    
if 1: #choose one below, both create Figure 5C parabola
    if 1:
        plt.scatter(Y1.T, Y2.T, s=1)
        plt.xlabel(r'${Y_1}^T$')
        plt.ylabel(r'${Y_2}^T$')
    if 0:
        plt.scatter(Y1, Y2, s=1)
        plt.xlabel(r'$Y_1$')
        plt.ylabel(r'$Y_2$')
    plt.show()
    
if 1: #cosine grid (3D)
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    ax.scatter(X0, Y1.T, Y1)
    ax.set_xlabel('$X_0$')
    ax.set_ylabel('${Y_1}^T$')
    ax.set_zlabel('$Y_1$')
    #ax.view_init(elev=90, azim=0) #sinusoid projection (i.e., 3D view of Figure 5A)
    ax.view_init(elev=0, azim=0) #Figure 5D
    ax.auto_scale_xyz
    plt.show()
    
if 1: #choose one below, both create Figure 5E parabolic surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')  
    if 1:
        ax.scatter(Y1.T, Y1, Y2.T)
        ax.set_xlabel('${Y_1}^T$')
        ax.set_ylabel('$Y_1$')
        ax.set_zlabel('${Y_2}^T$')
    else:
        ax.scatter(Y1, Y1.T, Y2)
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('${Y_1}^T$')
        ax.set_zlabel('$Y_2$')
    ax.view_init(elev=15, azim=-60)
    ax.auto_scale_xyz
    plt.show()