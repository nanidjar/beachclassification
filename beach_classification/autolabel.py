from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import classification_report
import os
import pickle
import numpy as np
from scipy.stats import mode

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import utm

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

class autolabel:

    def __init__(self, clf_path, survey_path, mode, parameters):
        if clf_path.endswith('/'):
            self.clf_path = clf_path
        else:
            self.clf_path = clf_path + '/'
        
        self.clf_path = clf_path
        self.survey_path = survey_path
        self.mode = mode
        self.parameters = parameters
        self.clf = None
        self.report = None
        self.labels = None
        
    def initialize_classifier(self):
        
        
        if self.mode.lower() != 'train':
            raise Exception("mode must be 'train' in order to use function 'initialize_classifier'")
        else:
            pass
        
        
        params = self.parameters

        path = self.clf_path

        clf_name = params["name"]
        report = 'analysis_report'
        test_data = 'testing'
        train_data = 'training'

        clfnew = path + '/' + clf_name

        report_folder = clfnew + '/' + report
        test_folder = clfnew + '/' + test_data
        train_folder = clfnew + '/' + train_data

        clf_folders = (clfnew, report_folder, test_folder, train_folder)

        for elem in clf_folders:
            try:
                os.mkdir(elem)
            except:
                pass

        if params['classifier'].lower() == 'gaussian process':
            clfr = GaussianProcessClassifier()
            
        elif params['classifier'].lower() == 'random forest':
            clfr = RandomForestClassifier()
            
        elif params['classifier'].lower() == 'mlp':
            clfr = MLPClassifier()
            
        else:
            raise Exception(
                "Classifier type must be {'Gaussian Process', 'Random Forest', or 'MLP'}")
        
        clf = GridSearchCV(clfr, params['clf_params'],scoring = 'balanced_accuracy')
        pipe = Pipeline([('scaler', StandardScaler()),
                             ('transformer', IncrementalPCA(n_components = 1, whiten = True)), ('learner', clf)])
        
        try:
            with open(clfnew  + '/init_clf.pickle','wb') as f:
                pickle.dump(pipe, f, pickle.HIGHEST_PROTOCOL)
        except:
            pass
        
        return pipe
    
    
    
    def train(self):
        
        
        if self.mode.lower() != 'train':
            raise Exception("mode must be 'train' in order to use function 'train'")
        else:
            pass
        
        params = self.parameters
        
        clf_name = params["name"]
        path = self.clf_path
        clf_path = path + clf_name + '/init_clf.pickle'
        clf = pickle.load( open( clf_path, "rb" ) )
        
        
        if type(self.survey_path) == str:
            features = np.load(self.survey_path + '/resolution' + str(params["resolution"]) + '/2_rawtexturefeatures/features.npy')
        
        elif type(self.survey_path) == list:
            
            feature_list = []
            
            for path in self.survey_path:
                features = np.load(path + '/resolution' + str(params["resolution"]) + '/2_rawtexturefeatures/features.npy')
                feature_list.append(features)

            features = np.concatenate(tuple(feature_list),axis = 0)
        
        clean_features = features[~np.isnan(features).any(axis=1)]
        cf = clean_features[clean_features[:,4] != 0]
        train_feats = np.delete(cf,[0,1,2,4],1)
        
        # print(cf.shape)
        # print("features shape:", features.shape)
        feat_train, feat_test, label_train, label_test = train_test_split( 
            np.nan_to_num(train_feats), np.nan_to_num(cf[:,4]), test_size=0.33, random_state=97)
        
        
        path = self.clf_path

        clf_name = params["name"]
        report = 'analysis_report'
        test_data = 'testing'
        train_data = 'training'

        clfnew = path + clf_name

        report_folder = clfnew + '/' + report
        test_folder = clfnew + '/' + test_data
        train_folder = clfnew + '/' + train_data
        
        try:
            np.savez(train_folder + '/training.npz' ,features = feat_train, labels = label_train)
            np.savez(test_folder + '/testing.npz' ,features = feat_test, labels = label_test)
        except:
            pass
        
        clf.fit(feat_train,label_train)
        
        self.clf = clf
        
        params = self.parameters
        path = self.clf_path
        clf_name = params["name"]
        clfnew = path + '/' + clf_name
        
        with open(clfnew  + '/trained_clf.pickle','wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        
        target_names = ['unclassified', 'water']
        self.report = classification_report(label_test, clf.predict(feat_test), output_dict = True, target_names = target_names)
    

    def autolabel(self):
        
        if self.mode != 'label':
            raise Exception("Mode must be 'label' in order to use function 'autolabel'")
        else:
            pass
        
        if type(self.survey_path) == list:
            raise Exception("Survey Path must be a string with the path to a single survey to automatically label")
        else:
            pass
        
        params = self.parameters
        
        clf_name = params["name"]
        path = self.clf_path
        clf_path = path + clf_name + '/trained_clf.pickle'
        clf = pickle.load( open( clf_path, "rb" ) )
        
        features = np.load(self.survey_path + '/resolution' + str(params["resolution"]) + '/2_rawtexturefeatures/features.npy')
        texture_feats = np.delete(features,[0,1,2,4],1)
        dirty_labels = clf.predict(texture_feats)
        
        if self.parameters['knn_clean']:
            neigh = NearestNeighbors(n_neighbors = self.parameters["k"])
            xy = np.transpose(np.vstack((features[:,0],features[:,1])))
            neigh.fit(xy)
            indi = neigh.kneighbors(xy, return_distance=False)

            labels = dirty_labels

            for i,ind in enumerate(indi):
                point = dirty_labels[ind]
                nlab, _ = mode(point)
                labels[i] = nlab[0]
        else:
            labels = dirty_labels
        
        self.labels = [features[:,0], features[:,1], labels]
        
        
    def visual_check(self, **kwargs):
        
        # filt = self.labels[2] == 1
        x, y, labels = self.labels
        # x = x[filt]
        # y = y[filt]
        # labels = labels[filt]
        coordmin = utm.to_latlon(y.min(), x.min(),11,northern = True)
        coordmean = utm.to_latlon(y.mean(), x.mean(),11,northern = True)
        coordmax = utm.to_latlon(y.max(), x.max(),11,northern = True)
        lat,lon = utm.to_latlon(y, x,11,northern = True)
        
        cMap = ListedColormap(['goldenrod','cornflowerblue'])
        #cMap.set_over('1')
        #cMap.set_under('2')
        
        fig0, ax0 = plt.subplots(1,1, figsize = [8,20])

        m = Basemap(projection='nsper', resolution='f', 
                    llcrnrlon=coordmin[1], llcrnrlat=coordmin[0],
                    urcrnrlon=coordmax[1],urcrnrlat=coordmax[0],
                    lat_0=coordmean[0], lon_0=coordmean[1],
                    width=1.0E3, height=2.4E3, epsg = 4269, ax = ax0)
        # m.shadedrelief()
        m.arcgisimage(service='World_Imagery', verbose= False)
        
        x,y = m(lon,lat)
        c0 = m.scatter(x,y,1,labels,latlon=True,cmap=cMap, **kwargs)
        
        plt.title('Applied Labels')
        # cbaxes = fig0.add_axes()
        cbar = fig0.colorbar(c0,ax=ax0 ,cmap = cMap, orientation = 'horizontal', pad = 0.03, shrink = 0.5)
        cbar.ax.set_xlabel('Label',labelpad = 30, rotation=0)
        cbar.ax.invert_xaxis()
        cbar.ax.get_xaxis().set_ticks([])
        cbar.ax.text(1.25, 0.25, 'land', ha='center', va='center')
        cbar.ax.text(1.75, 0.25, 'water', ha='center', va='center')
        
        parallels = np.arange(lat.min(),lat.max(),0.005)
        # labels = [left,right,top,bottom]
        m.drawparallels(parallels,labels=[False,True,True,False],linewidth = 2)
        meridians = np.arange(lon.min(),lon.max(),0.005)
        m.drawmeridians(meridians,labels=[True,False,False,True],linewidth = 2)
        
    def apply(self):
        
        histpath = self.survey_path + '/resolution' + str(self.parameters["resolution"]) + '/1_histogram/'
        
        hist = np.load(histpath + 'hist.npz')
        
        X = np.vstack((self.labels[0],self.labels[1])).transpose()
        
        labels = griddata(X, self.labels[2], (hist['x'],hist['y']), method = 'nearest')
        
        label_mask = ~np.isnan(hist['labels'])
        masked = labels * label_mask.astype(int)
        masked[masked ==0] = np.nan
        
        np.save(histpath + 'labels.npy', masked)
        
        
        
        
        
