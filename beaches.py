import scipy
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

import matplotlib.pyplot as plt
from numba import jit
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA

def gaborkernels(num_orientations=int(4), bandwidths=(1, 3), frequencies=(0.125, 0.25)):

    kernels = []

    # loop through kernel orientations
    for theta in range(num_orientations):
        theta = theta / num_orientations * np.pi

        # loop through bandwidths
        for sigma in bandwidths:

            # loop through frequencies
            for frequency in frequencies:

                # calculate and take the real part of a gabor wavelet
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))

                # append to kernel list
                kernels.append(kernel)

    return kernels


def blockfeats(attr, kernels, x, y, z, I, labels, blocksize=20, step=5):
    # """Function to calculate mean and variance features
    # of blocks in an image filtered using gabor wavelet kernels."""

    def compute_feats(image, kernels):
        # """Function to calculate the features of an image using a bank of convolution kernels.

        # The function takes two inputs.
        # 'image' is a 2D numpy array.
        # 'kernels' is a list of 2D numpy arrays

        # The output will be a numpy array of shape (len(kernels) x 2) where the first column
        # contains the mean of each filtered image, and the second column contains the
        # variance of each filtered image."""

        # replace nans with zeros
        imagenan = np.isnan(image)
        image[imagenan] = 0

        # create the feaeture array, one column for mean, and one for variance.
        feats = np.zeros((2,len(kernels)), dtype=np.double)

        # loop through the gabor wavelet kernels
        for k, kernel in enumerate(kernels):

            # filter the image using the gabor wavelet
            filtered = ndi.convolve(image, kernel, mode='wrap')

            # calculate mean and variance of the filtered image, and add each to the feature array
            feats[0, k] = np.mean(filtered)
            feats[1, k] = np.var(filtered)

        return feats

    # set the size of each image block (in pixels) to filter with the gabor kernels.

    # half block size to get indices of block around a pixel.
    hb = int(blocksize/2)

    # step size between blocks (if step < blocksize, blocks will be overlapping)

    # initialize numpy arrays for mean and var of each filtered block

    num_samples = np.floor(
        ((attr.shape[0]-(2*hb))/step))*np.floor(((attr.shape[1]-(2*hb))/step))
    
    
    numkern = len(kernels)

    # kernels + x , y, z, I, labels
    attr_mean = np.zeros((num_samples.astype(int), numkern+5))
    attr_var = np.zeros_like(attr_mean)

    # loop through pixels, starting at half a blocksize away from the edge ( to avoid padding)
    # and incremented by step.
    i = 0
    for xi in range(hb, attr.shape[0]-hb, step):
        for yi in range(hb, attr.shape[1]-hb, step):

            # if the block has no data in it, just assign the mean and var of that block to zero
            if np.isnan(attr[xi, yi]):
                attr_mean[i, 5:] = np.nan
                
                attr_var[i, 5:] = np.nan

            # calculate features and add mean and var to corresponding numpy arrays.
            else:
                coldfeat = compute_feats(
                    attr[xi-hb:xi+hb, yi-hb:yi+hb], kernels)

                attr_mean[i, 5:] = coldfeat[0, :]
                attr_var[i, 5:] = coldfeat[1, :]

            attr_mean[i, 0] = x[xi, yi]
            attr_mean[i, 1] = y[xi, yi]
            attr_mean[i, 2] = z[xi, yi]
            attr_mean[i, 3] = I[xi, yi]
            attr_mean[i, 4] = labels[xi, yi]

            attr_var[i, :5] = attr_mean[i, numkern:]

            i += 1

    return attr_mean, attr_var


def texturePCA(attribute, ncomp=3):

    # remove zeros
    kernattr = attribute[np.any(attribute[:, 6:], axis=1),:]
    kernattr = kernattr[np.any(~np.isnan(kernattr), axis=1),:]

    # create PCA transformer
    EOF = IncrementalPCA(n_components=ncomp, batch_size=200)

    # normalize kernel feats
    kernelfeats = np.nan_to_num(StandardScaler().fit_transform(kernattr))

    # fit kernel attributes
    EOF.fit(kernelfeats)

    # get the PCAs
    pcas = EOF.transform(kernelfeats)

    ncol = int(5+ncomp)

    reduced_attr = np.zeros((kernattr.shape[0], ncol))

    reduced_attr[:, :5] = kernattr[:, :5]

    reduced_attr[:, -ncomp:] = pcas

    return reduced_attr, EOF

def full_preprocess(delmar, featFilename):
    
    np.seterr(invalid='ignore')
    
    x, y = np.meshgrid(delmar['y'], delmar['x'])
    z = delmar['zsum']/delmar['count']
    I = delmar['isum']/delmar['count']
    
    
    labels = delmar['labelsum']/delmar['count']
    water = labels >= 4.5 # water
    unclass = labels < 4.5 # unclassified
    
    # for troubleshooting labels with value of zero around accurately classified water points
    labels[water] = 2
    labels[unclass] = 1

    kerns = gaborkernels()

    intensityfeats = blockfeats(z, kerns, x, y, z, I, labels, 20, 5)
    
    np.save(featFilename, intensityfeats)
    
    featurevecs, transformer = texturePCA(intensityfeats[0])
    
    return featurevecs, transformer