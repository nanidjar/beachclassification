import laspy
import scipy
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi

from numba import jit
import pickle
import os
from os import path
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from IPython.display import clear_output


class survey:

    def __init__(self, filepath, parameters, outputpath):
        
        self.filepath = filepath
        
        if type(filepath) == str:
            if  filepath.endswith('.las'):
                pass
            else:
                raise Exception("filepath must be a path to a file ending with .las")

            if path.isfile(filepath):
                pass
            else:
                raise Exception("filepath does not exist")
        elif type(filepath) == list:
            pass
        else:
            raise Exception("filepath must be a string")
            
        
            
        

        self.parameters = parameters
        
        if type(parameters) == dict:
            pass
        else:
            raise Exception("parameters must be of type 'dict'")
            
            
            
        
        self.outputpath = outputpath
        
        if type(outputpath) == str:
            pass
        else:
            raise Exception("outputpath must be a string")
        

        if outputpath.endswith('/'):
            pass
        else:
            self.outputpath += '/'
            
        if path.isdir(outputpath):
            pass
        else:
            raise Exception("outputpath does not exist")

        self.features = None
        self.histpath = None
        self.featpath = None

##########################################################################################
       
    def create_dir(self):

        binsize = self.parameters["resolution"]
        folders_main = self.outputpath
        survey_name = self.parameters["survey_name"]

        raw = 'raw'
        resolution = 'resolution'
        h = '1_histogram/'
        txtr = '2_rawtexturefeatures/'
        res_value = str(binsize)

        survey_folder = folders_main + survey_name

        raw_survey = survey_folder + '/' + raw
        res_survey = survey_folder + '/' + resolution + res_value

        hist = res_survey + '/' + h 
        txtfeat = res_survey + '/' + txtr

        allfolders = (survey_folder, raw_survey, res_survey, hist, txtfeat)

        for elem in allfolders:
            try:
                os.mkdir(elem)
            except:
                continue
                
        self.histpath = hist
        self.featpath = txtfeat

##################################################################################

    # @jit(parallel=True)
    def downsample(self):
        
        if path.isfile(self.histpath + 'hist.npz'):
            raise Exception("This survey has already been downsampled!")
            
        inFile = laspy.file.File(self.filepath, mode='r')
        num_points = inFile.__len__()
        scale = inFile.header.scale
        offset = inFile.header.offset
        xmax, ymax, _ = inFile.header.max
        xmin, ymin, _ = inFile.header.min
        binsize = 1

        xbin = np.arange(np.floor(xmin), np.ceil(xmax), binsize)
        ybin = np.arange(np.floor(ymin), np.ceil(ymax), binsize)
        count = np.zeros((xbin.shape[0], ybin.shape[0]))
        zsum = np.zeros((xbin.shape[0], ybin.shape[0]))
        Itsum = np.zeros((xbin.shape[0], ybin.shape[0]))
        labelsum = np.zeros((xbin.shape[0], ybin.shape[0]))

        x_gen = (x*scale[0] + offset[0] for x in inFile.X)
        y_gen = (y*scale[1] + offset[1] for y in inFile.Y)
        z_gen = (z*scale[2] + offset[2] for z in inFile.Z)
        I_gen = (I for I in inFile.intensity)
        label_gen = (label for label in inFile.raw_classification)

        for i,x, y, z, I, label in zip(range(num_points),x_gen, y_gen, z_gen, I_gen, label_gen):
            if ~(label == 7) & ~(label == 18):  # ignore noise values
                xi = np.digitize(x, xbin)
                yi = np.digitize(y, ybin)
                count[xi-1, yi-1] += 1
                zsum[xi-1, yi-1] += z
                Itsum[xi-1, yi-1] += I
                labelsum[xi-1, yi-1] += label
            else:
                continue

            #clear_output(wait = True)
            #print('Downsampling: ', (i/num_points) * 100, '%',flush = True)

        np.seterr(invalid='ignore')

        x, y = np.meshgrid(ybin, xbin)
        z = zsum / count
        I = Itsum / count
        labels = labelsum / count

        # original labels: water is 9, unclassified is 0, reef is 26
        water = labels >= 4.5
        unclass = labels < 4.5
        labels[unclass] = 1
        labels[water] = 2

        histpath = self.outputpath + self.parameters["survey_name"] + '/resolution' + str(
            self.parameters["resolution"]) + '/1_histogram/' + 'hist.npz'
        np.savez(histpath, x=x, y=y, count=count, z=z, I=I, labels=labels)

        clear_output(wait = True)
        print('Downsampling: 100 %',flush = True)

########################################################################################
  
    def texture(self):
        
        if path.isfile(self.featpath + 'features.npy'):
            raise Exception("Texture has already been derived for this survey!")
        ##########################################################

        num_orientations = self.parameters["gabor_orient"]
        bandwidths = self.parameters["gabor_bw"]
        frequencies = self.parameters["gabor_freq"]

        kernels = []

        # loop through kernel orientations
        for theta in range(int(num_orientations)):
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

        ###########################################################

        histspath = self.outputpath + self.parameters["survey_name"] + '/resolution' + str(
            self.parameters["resolution"]) + '/1_histogram/' + 'hist' + '.npz'

        hist = np.load(histspath)

        
        if self.parameters["texture_attr"].lower() == 'height':
            attr = hist['z']
        elif self.parameters["texture_attr"].lower() == 'intensity':
            attr = hist['I']
        else:
            attr = hist['z']

        ###############################################################

        block_size = self.parameters["block_size"]
        step_size = self.parameters["step_size"]

        hb = int(block_size/2)

        ##################################################################

        # num_samples = np.floor(
        #    ((attr.shape[0] - (2*hb))/step_size) + 1) * np.floor(((attr.shape[1] - (2*hb))/step_size) + 1)

        num_samples = np.arange(
            hb, attr.shape[0], step_size).shape[0] * np.arange(hb, attr.shape[1], step_size).shape[0]

        numkern = len(kernels)

        # kernels + x , y, z, I, labels

        features = np.zeros((num_samples, numkern+5))

        ###################################################################
        i = 0
        for xi in range(hb, attr.shape[0]-hb, step_size):
            for yi in range(hb, attr.shape[1]-hb, step_size):

                if np.isnan(attr[xi, yi]):
                    features[i, :] = np.nan
                else:
                    block = attr[xi-hb:xi+hb, yi-hb:yi+hb]
                    features[i, [0, 1, 2, 3, 4]] = [hist['x'][xi, yi], hist['y'][xi, yi],
                                                    hist['z'][xi, yi], hist['I'][xi, yi], hist['labels'][xi, yi]]

                    for kernel in kernels:
                        #print('kernel shape is : ', kernel.shape)
                        #print('block shape is : ', block.shape)
                        feature = np.mean(ndi.convolve(
                            block, kernel, mode='wrap'))
                        for j in range(5, len(kernels) + 5):
                            features[i, j] = feature

                clear_output(wait=True)
                print('Deriving texture attributes: ', round(
                    (i/num_samples) * 100, 2), '%', flush=True)
                i += 1

        ########################################################################

        clean_features = features[~np.isnan(features).any(axis=1)]
        cf = clean_features[clean_features[:, 4] != 0]

        self.features = cf

        txtrpath = self.outputpath + self.parameters["survey_name"] + '/resolution' + str(
            self.parameters["resolution"]) + '/2_rawtexturefeatures/' + 'features' + '.npy'

        np.save(txtrpath, cf)

        clear_output(wait = True)
        print('Deriving texture attributes: 100 %', flush=True)
        
        
    def plot(self,attribute, **kwargs):
        
        if path.isfile(self.histpath + 'hist.npz'):
            pass
        else:
            raise Exception("histogram has not yet been created.")
            
        hist = np.load(self.histpath + 'hist.npz')
        
        if type(attribute) is str:
            pass
        else:
            raise Exception("attribute must be of type 'string'")
        
        
        fig0=plt.figure(figsize=[8,5])
        
        
        if attribute.lower() == 'height':
            
            ax0 = fig0.add_subplot(111)
            c0 = plt.pcolor(hist['x'],hist['y'],hist['z'], **kwargs)
            plt.title(self.parameters["survey_name"])
            cbar = fig0.colorbar(c0,ax=ax0)
            cbar.ax.set_ylabel('Height (m)', rotation=-90, va="bottom")
            
        elif attribute.lower() == 'intensity':
            
            ax0 = fig0.add_subplot(111)
            c0 = plt.pcolor(hist['x'],hist['y'],hist['I'], **kwargs)
            plt.title(self.parameters["survey_name"])
            cbar = fig0.colorbar(c0,ax=ax0)
            cbar.ax.set_ylabel('Return Intensity', rotation=-90, va="bottom")
            
        elif attribute.lower() == 'classification':
            if path.isfile(self.histpath + 'labels.npy'):
                # plot with predicted labels
                pass
            else:
                
                ax0 = fig0.add_subplot(111)
                c0 = plt.pcolor(hist['x'],hist['y'],hist['labels'], **kwargs)
                plt.title(self.parameters["survey_name"])
                cbar = fig0.colorbar(c0,ax=ax0)
                cbar.ax.set_ylabel('raw classification', rotation=-90, va="bottom")
        