import os, os.path, sys
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
from pylab import imshow, show, loadtxt
from scipy import stats

# ====================================================================================
# Generate noise on a given image (run via external scripts) 
# Author:    E. Seitz @ Columbia University - Frank Lab - 2020
# Contact:   evan.e.seitz@gmail.com
# ====================================================================================

pyDir = os.path.dirname(os.path.abspath(__file__)) #python file directory

# function to find SNR:
def find_SNR(image, SNR):
    IMG_2D = []
    for pix in image:
        IMG_2D.append(pix)

    IMG_1D = list(itertools.chain.from_iterable(IMG_2D))
    img_mean = np.mean(IMG_1D)
    img_var = np.var(IMG_1D)
    img_std = np.sqrt(img_var)
    
    SIG_1D = []
    for pix in IMG_1D:
        if -img_std*.25 < pix < img_std*.25:
            pass
        else:
            SIG_1D.append(pix) #only grab signal

    sig_mean = np.mean(SIG_1D)
    sig_var = np.var(SIG_1D)
    noise_var = sig_var / SNR #experimental regime
    noise_std = np.sqrt(noise_var)
    
    return sig_mean, noise_std

##########################
# function to add noise:
def add_noise(mean, std, image):
    row, col = image.shape
    gauss = np.random.normal(mean, std, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy

##########################
# function to standardize:
def standardize(image, check):
    bg = []
    h, w = image.shape[:2]
    center = [int(w/2), int(h/2)]
    radius = int(h/2)
    Y, X = np.ogrid[:h,:w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = dist_from_center <= radius
    masked_img = image.copy()
    if 0: #for visualization of mask only, keep off (0)
        masked_img[~mask] = 0
        plt.imshow(masked_img, cmap='gray')
        plt.show()
    bg = masked_img[~mask]
    bg_mean = np.mean(bg)
    bg_std = np.std(bg)
    if bg_std == 0:
        img_norm = 0
    else:
        img_norm = (image - bg_mean) / bg_std
    if check is True: #for testing only
        print('bg_mean:', bg_mean)
        print('bg_std:', bg_std)
    return img_norm

def op(img_orig, SNR): #experimental SNR ~ 0.1
    
    sig_mean, noise_std = find_SNR(img_orig, SNR) #find SNR
    img_noise = add_noise(sig_mean, noise_std, img_orig) #apply noise
    img_norm = standardize(img_noise, check=False) #standardize
    
    if 0: #print out new mean and standard deviation: {~0,1}
        img_norm_check = standardize(img_norm, check=True)
    
    return img_norm
        

        


