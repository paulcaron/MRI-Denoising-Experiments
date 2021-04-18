"""
Given precomputed volumes of central, average and denoised images, 
compute the PSNR and save images for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from nibabel.testing import data_path

from sklearn import linear_model
from sklearn.neural_network import MLPRegressor

import h5py

def psnr(x_hat, x_true, maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse = np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

from dipy.io.image import save_nifti, load_nifti
from dipy.io import read_bvals_bvecs

N_CLUSTERS = 15 # update this parameter


if __name__ == "__main__":
        
    f_central = h5py.File('../central_images_'+str(N_CLUSTERS)+'_clusters.h5', 'r')
    f_denoised = h5py.File('../denoised_'+str(N_CLUSTERS)+'_clusters.h5', 'r')
    f_average = h5py.File('../averages_'+str(N_CLUSTERS)+'_clusters.h5', 'r')
    
    
    central_data = f_central['dataset_1'].value
    denoised_data = f_denoised['dataset_1'].value
    average_data = f_average['dataset_1'].value
   
    print("PSNR for Central Data:", psnr(central_data, average_data))
    print("PSNR for Denoised Data:", psnr(denoised_data, average_data))
    
    for i in range(N_CLUSTERS):
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_average_"+str(i)+".png", average_data[:,:,50,i], cmap='gray')
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_central_"+str(i)+".png", central_data[:,:,50,i], cmap='gray')
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_denoised_"+str(i)+".png", denoised_data[:,:,50,i], cmap='gray')
