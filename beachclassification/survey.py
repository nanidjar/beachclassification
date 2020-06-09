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
import yaml

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from IPython.display import clear_output

from matplotlib.colors import ListedColormap
from mpl_toolkits.basemap import Basemap
import utm
from tqdm.notebook import tqdm

class survey:
    """Example usage:
        
        output_dir = 'C:/Users/user1/Desktop/Beach'
        survey_path = 'C:/path/to/survey.las'
        
        test_survey = survey(output_dir,survey_path)
        test_survey.resample()
        test_survey.texture()
        
        %matplotlib notebook
        
        test_survey.plot('intensity')
        
        test_survey.plot('height')
        
        test_survey.plot('classification')
        """

    def __init__(self, outputpath, filepath,  **params):
        """Inputs:
        
        outputpath : str 
            The path to a folder where you will store all the surveys processed with a
            specified parameter set.
        
        filepath : str
            The path to a .las file
                OR
            The unique name of a survey folder in 'outputpath'. 
            
        params: *optional*
            resolution: float
                The desired resampling resolution in meters. (default is 0.5)
            block_size: float
                Half the side length in meters of a square window centered around a sample point
                for deriving texture attributes. (default is 3)
            step_size: float
                Distance in meters between each sample point around which to derive texture
                attributes (default is 2)
            gabor_orient: int
                The number of gabor wavelet orientations with which to filter samples
                to derive texure attributes. (default is 4)
            gabor_freq: tuple
                A tuple of spatial frequencies in cycles per meter, specifying the gabor wavelets 
                to be used for deriving texture attributes. Keep the nyquist frequency in mind 
                (half the sample rate). (default is (0.125,0.25))
            gabor_bw: tuple
                A tuple of bandwidths, specifying the gabor wavelets to be used for deriving 
                texture attributes. (default is (1,3))
            texture_attr: str, 'height' or 'intensity'
                which point attribute to use for deriving texture attributes. (default is 'height')
        """
        
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
            
            
            
            
            
        self.filepath = filepath
        
        if type(filepath) == str:
            if path.isfile(filepath):
                if  filepath.endswith('.las'):
                    filename = path.basename(filepath)
                    name = filename.split('.',1)[0]
                    self.surveyname = name
                else:
                    raise Exception("filepath must be a path to a file ending with .las")
            elif path.isdir(self.outputpath + filepath):
                self.surveyname = filepath
            else:
                raise Exception("The survey cannot be found!")
        elif filepath == None:
            pass
        else:
            raise Exception("filepath must be a string")
            
        
            
        self.params = params
        self._config_file()
        
        
        if type(self.parameters) == dict:
            pass
        else:
            raise Exception("parameters must be of type 'dict'")

        self.features = None
        self.histpath = None
        self.featpath = None
        
        self._create_dir()
        

##########################################################################################
    def _config_file(self):
        """
        Adds a configuration file 'config.yaml' to the output directory given by the user.
        """
        
        config_path = self.outputpath + 'config.yaml'
        
        if path.isfile(config_path):
            with open(config_path, 'r') as f:
                parameters = yaml.load(f)
                self.parameters = parameters
            if len(self.params) > 0:
                raise Exception('Cannot change parameters in existing configuration file. This ensures consistency among processed surveys in a folder. Create a new folder if you would like to experiment with new parameters.')
        else:
            parameters = {

                "resolution" : 0.5,
                "gabor_orient" : 4,
                "gabor_bw" : (1,3),
                "gabor_freq" : (0.125, 0.25),

                "block_size" : 3,
                "step_size" : 2,
                "texture_attr" : 'height' # or 'intensity'
            }   

            parameters.update(self.params)
            
            for fr in parameters["gabor_freq"]:
                if fr > (0.5 / parameters["resolution"]):
                    raise Exception("Gabor wavelet frequencies must be smaller than the nyquist frequency (half the sampling rate)")
                else:
                    continue
                                    
                                     
            self.parameters = parameters
            
            with open(config_path, 'w') as f:
                yaml.dump(parameters,f)
            
###############################################################################################
                    
    def _create_dir(self):
        """
        Creates the following directory tree.
        
        survey
        |--raw
        |__resolution[value]
             |--1_histogram
             |__2_rawtexturefeatures
        
        """
        
        binsize = self.parameters["resolution"]
        folders_main = self.outputpath
        survey_name = self.surveyname

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
    def resample(self):
        """Uniformly resamples the survey to the resolution in the configuration file. 
        
        Concession: This function for resampling does not lowpass filter the data before
        resampling. For this reason, there may be artifacts such as moire patterns in the 
        resampled data.
        """
        
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
        
        loop_length = np.sum(inFile.header.point_return_count)

        for i,x, y, z, I, label in tqdm(zip(range(num_points),x_gen, y_gen, z_gen, I_gen, label_gen),total = loop_length):
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

        histpath = self.outputpath + self.surveyname + '/resolution' + str(
            self.parameters["resolution"]) + '/1_histogram/' + 'hist.npz'
        np.savez(histpath, x=x, y=y, count=count, z=z, I=I, labels=labels)

        clear_output(wait = True)
        print('Downsampling: 100 %',flush = True)

########################################################################################
     
    def texture(self):
        """Derives texture attributes in a window around a sample of points in the survey.
        """
        
        if path.isfile(self.featpath + 'features.npy'):
            raise Exception("Texture has already been derived for this survey!")
            
        if path.isfile(self.histpath + 'hist.npz'):
            pass
        else:
            raise Exception("Texture can only be derived AFTER downsampling using survey.downsample()")
        ##########################################################

        num_orientations = self.parameters["gabor_orient"]
        bandwidths = self.parameters["gabor_bw"]
        frequencies = self.parameters["gabor_freq"]/self.parameters["resolution"]
        
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

        histspath = self.outputpath + self.surveyname + '/resolution' + str(
            self.parameters["resolution"]) + '/1_histogram/' + 'hist' + '.npz'

        hist = np.load(histspath)

        
        if self.parameters["texture_attr"].lower() == 'height':
            attr = hist['z']
        elif self.parameters["texture_attr"].lower() == 'intensity':
            attr = hist['I']
        else:
            attr = hist['z']

        ###############################################################

        block_size = int(self.parameters["block_size"]/self.parameters["resolution"])
        step_size = int(self.parameters["step_size"]/self.parameters["resolution"])

        hb = block_size

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
        for xi in tqdm(range(hb, attr.shape[0]-hb, step_size)):
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

                #clear_output(wait=True)
                #print('Deriving texture attributes: ', round(
                #    (i/num_samples) * 100, 2), '%', flush=True)
                i += 1

        ########################################################################

        clean_features = features[~np.isnan(features).any(axis=1)]
        cf = clean_features[clean_features[:, 4] != 0]

        self.features = cf

        txtrpath = self.outputpath + self.surveyname + '/resolution' + str(
            self.parameters["resolution"]) + '/2_rawtexturefeatures/' + 'features' + '.npy'

        np.save(txtrpath, cf)

        clear_output(wait = True)
        print('Deriving texture attributes: 100 %', flush=True)
        
        
    def plot(self,attribute, **kwargs):
        """Plots the resampled survey over a map.
        
        The following attributes can be plotted 'height', 'intensity', 'classification'.
        
        If plotting 'classification', this method will plot either the raw classification 
        given in the .las file, or if it has been labelled by a trained classifier using 
        the autolabel class, this method will plot these automatic labels. 
        
        Inputs:
            attribute: str, 'height', 'intensity', 'classification'
                attribute to plot.
            kwargs: *optional*
                Any optional arguments to matplotlib.pyplot.pcolormesh can be passed here to
                customize the plot. for example, 'vmin' and 'vmax' will set the colorbar limits. 
        """
        
        if path.isfile(self.histpath + 'hist.npz'):
            pass
        else:
            raise Exception("histogram has not yet been created.")
            
        hist = np.load(self.histpath + 'hist.npz')
        
        if type(attribute) is str:
            pass
        else:
            raise Exception("attribute must be of type 'string'")
        
        coordmin = utm.to_latlon(hist['y'].min(),hist['x'].min(),11,northern = True)
        coordmean = utm.to_latlon(hist['y'].mean(),hist['x'].mean(),11,northern = True)
        coordmax = utm.to_latlon(hist['y'].max(),hist['x'].max(),11,northern = True)
        lat,lon = utm.to_latlon(hist['y'].transpose(),hist['x'].transpose(),11,northern = True)
        
        cMap = ListedColormap(['goldenrod','cornflowerblue'])
        cMap.set_over('1')
        cMap.set_under('2')
        
        # 1. Draw the map background
        fig0, ax0 = plt.subplots(1,1, figsize = [8,20])

        m = Basemap(projection='nsper', resolution='f', 
                    llcrnrlon=coordmin[1], llcrnrlat=coordmin[0],
                    urcrnrlon=coordmax[1],urcrnrlat=coordmax[0],
                    lat_0=coordmean[0], lon_0=coordmean[1],
                    width=1.0E3, height=2.4E3, epsg = 4269, ax = ax0)
        
        m.arcgisimage(service='World_Imagery', verbose= False)
        
        
        
        
        if attribute.lower() == 'height':
            
            c0 = m.pcolormesh(lon,lat,hist['z'].transpose(),latlon=True,cmap='gist_rainbow', **kwargs)
            plt.title(self.surveyname)
            cbaxes = fig0.add_axes()
            cbar = fig0.colorbar(c0,ax=cbaxes,cmap = cMap, orientation = 'horizontal', pad = 0.03, shrink = 0.5)
            cbar.ax.set_xlabel('NAVD 88 Height (m)',labelpad = 30, rotation=0)
        elif attribute.lower() == 'intensity':
            
            
            c0 = m.pcolormesh(lon,lat,hist['I'].transpose(),latlon=True,cmap='gist_rainbow', **kwargs)
            plt.title(self.surveyname)
            cbaxes = fig0.add_axes()
            cbar = fig0.colorbar(c0,ax=cbaxes,cmap = cMap, orientation = 'horizontal', pad = 0.03, shrink = 0.5)
            cbar.ax.set_xlabel('Return Intensity',labelpad = 30, rotation=0)
            
        elif attribute.lower() == 'classification':
            if path.isfile(self.histpath + 'labels.npy'):
                
                labels = np.load(self.histpath + 'labels.npy')
                c0 = m.pcolormesh(lon,lat,labels.transpose(),latlon=True,cmap=cMap, **kwargs)
                plt.title(self.surveyname)
                cbaxes = fig0.add_axes()
                cbar = fig0.colorbar(c0,ax=cbaxes,cmap = cMap, orientation = 'horizontal', pad = 0.03, shrink = 0.5)
                cbar.ax.set_xlabel('Automatic Labels',labelpad = 30, rotation=0)
                cbar.ax.invert_xaxis()
                cbar.ax.get_xaxis().set_ticks([])
                cbar.ax.text(1.25, 0.25, 'land', ha='center', va='center')
                cbar.ax.text(1.75, 0.25, 'water', ha='center', va='center')
            else:
                
                c0 = m.pcolormesh(lon,lat,hist['labels'].transpose(),latlon=True,cmap=cMap)
                plt.title(self.surveyname)
                cbaxes = fig0.add_axes()
                cbar = fig0.colorbar(c0,ax=cbaxes,cmap = cMap, orientation = 'horizontal', pad = 0.03, shrink = 0.5)
                cbar.ax.set_xlabel('Manual Labels',labelpad = 30, rotation=0)
                cbar.ax.invert_xaxis()
                cbar.ax.get_xaxis().set_ticks([])
                cbar.ax.text(1.25, 0.25, 'land', ha='center', va='center')
                cbar.ax.text(1.75, 0.25, 'water', ha='center', va='center')
                
        parallels = np.arange(lat.min(),lat.max(),0.005)
        
        m.drawparallels(parallels,labels=[False,True,True,False],linewidth = 2)
        meridians = np.arange(lon.min(),lon.max(),0.005)
        m.drawmeridians(meridians,labels=[True,False,False,True],linewidth = 2)
        
