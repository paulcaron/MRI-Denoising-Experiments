"""
Given a number of clusters n:
	* Compute n clusters
	* For each cluster of size S, keep one "testing example" and keep S-1 "training examples"
	* Compute the average of the S-1 images as a "baseline"
	* Apply Patch2Self to the n testing examples
"""

from compute_PSNR import *
from patch2self import *

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import dipy
import h5py
from dipy.io import read_bvals_bvecs
import nibabel as nib
from sklearn.cluster import KMeans

N_CLUSTERS = 15 # update this parameter

if __name__ == "__main__":
    fdwi1 = "../MRIdenoising/Datab1000_session1.nii"
    fbval1 = "../MRIdenoising/Bval_session1.bval"
    fbvec1 = "../MRIdenoising/Bvec_session1.bvec"

    fdwi2 = "../MRIdenoising/Datab1000_session2.nii"
    fbval2 = "../MRIdenoising/Bval_session2.bval"
    fbvec2 = "../MRIdenoising/Bvec_session2.bvec"

    fdwi3 = "../MRIdenoising/Datab1000_session3.nii"
    fbval3 = "../MRIdenoising/Bval_session3.bval"
    fbvec3 = "../MRIdenoising/Bvec_session3.bvec"

    bvals1, bvecs1 = read_bvals_bvecs(fbval1, fbvec1)
    bvals2, bvecs2 = read_bvals_bvecs(fbval2, fbvec2)
    bvals3, bvecs3 = read_bvals_bvecs(fbval3, fbvec3)
    bvals = np.concatenate([bvals1, bvals2, bvals3])
    bvecs = np.concatenate([bvecs1, bvecs2, bvecs3])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(bvecs[:, 0], bvecs[:, 1], bvecs[:, 2])

    img1 = nib.load(fdwi1)
    img2 = nib.load(fdwi2)
    img3 = nib.load(fdwi3)

    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0).fit(bvecs)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2])

    clusters = kmeans.predict(bvecs)

    # plt.hist(clusters, bins=np.arange(29))
    # plt.show()

    images_cluster = []
    for cluster in range(N_CLUSTERS):
      l = []
      for i in range(len(clusters)):
        if clusters[i]==cluster:
          l.append(i)
      images_cluster.append(l)

    centrals = []
    for cluster in range(15):
      centroid = kmeans.cluster_centers_[cluster]
      points = np.array([bvecs[j] for j in images_cluster[cluster]])
      distances = np.linalg.norm(points-centroid)
      amin = distances.argmin()
      centrals.append(images_cluster[cluster][amin])

    central_images = np.zeros((292, 288, 192, N_CLUSTERS))
    print("Calculate central_images")
    for cluster in range(N_CLUSTERS):
      print(cluster)
      i = centrals[cluster]
      if i < 312:
        central_images[:,:,:,cluster] = img1.dataobj[:,:,:,i]
      elif i<2*312:
        central_images[:,:,:,cluster] = img2.dataobj[:,:,:,i-312]
      else:
        central_images[:,:,:,cluster] = img3.dataobj[:,:,:,i-2*312]

    averages = np.zeros((292, 288, 192, N_CLUSTERS))
    print("Calculate averages")
    for cluster in range(N_CLUSTERS):
      print(cluster)
      for i in images_cluster[cluster]:
        if i!=centrals[cluster]:
          if i < 312:
            averages[:,:,:,cluster] += img1.dataobj[:,:,:,i]
          elif i<2*312:
            averages[:,:,:,cluster] += img2.dataobj[:,:,:,i-312]
          else:
            averages[:,:,:,cluster] += img3.dataobj[:,:,:,i-2*312]
        averages[:,:,:,cluster] /= len(images_cluster[cluster])-1

    m = np.min([central_images.min(), averages.min()])
    M = np.max([central_images.max(), averages.max()])

    central_images = (central_images - m)/(M - m)
    averages = (averages - m)/(M - m)

    hf = h5py.File('../central_images_'+str(N_CLUSTERS)+'_clusters.h5', 'w')
    hf.create_dataset('dataset_1', data=central_images)
    hf.close()

    hf = h5py.File('../averages_'+str(N_CLUSTERS)+'_clusters.h5', 'w')
    hf.create_dataset('dataset_1', data=averages)
    hf.close()

    s3s_p2s = patch2self(central_images)

    del central_images
    del averages

    h5f = h5py.File('../denoised_'+str(N_CLUSTERS)+'_clusters.h5', 'w')
    h5f.create_dataset('dataset_1', data=s3s_p2s)
    h5f.close()

    del s3s_p2s

    f_central = h5py.File('../central_images_'+str(N_CLUSTERS)+'_clusters.h5', 'r')
    f_denoised = h5py.File('../denoised_'+str(N_CLUSTERS)+'_clusters.h5', 'r')
    f_average = h5py.File('../averages_'+str(N_CLUSTERS)+'_clusters.h5', 'r')

    central_data = f_central['dataset_1'].value
    denoised_data = f_denoised['dataset_1'].value
    average_data = f_average['dataset_1'].value

    print("central_data[:3, :3, 0, 0]", central_data[:3, :3, 0, 0])
    print("denoised_data[:3, :3, 0, 0]", denoised_data[:3, :3, 0, 0])
    print("average_data[:3, :3, 0, 0]", average_data[:3, :3, 0, 0])

    print("PSNR for Central Data:", psnr(central_data, average_data))
    print("PSNR for Denoised Data:", psnr(denoised_data, average_data))

    for i in range(N_CLUSTERS):
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_average_"+str(i)+".png", average_data[:,:,50,i], cmap='gray')
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_central_"+str(i)+".png", central_data[:,:,50,i], cmap='gray')
        plt.imsave("../average_P2S_"+str(N_CLUSTERS)+"_clusters/img_denoised_"+str(i)+".png", denoised_data[:,:,50,i], cmap='gray')
