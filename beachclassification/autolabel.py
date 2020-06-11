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
import yaml

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import utm

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap

class autolabel:
    """Example usage:
    
        Training:
    
            clf_path = "C:/Users/user1/Desktop/classifiers/"
            survey_list = ['C:/path/to/survey1', 'C:/path/to/survey2', 'C:/path/to/surveyn',]
            mode = 'train'
            name = 'test_clf'

            test_clf = autolabel(clf_path,survey_list, mode, name)
            test_clf.train()
            test_clf.report
            
        Labeling:
        
            %matplotlib notebook
        
            clf_path = "C:/Users/user1/Desktop/classifiers/"
            survey_path = 'C:/path/to/survey'
            mode = 'label'
            name = 'test_clf'
            
            trained_clf = autolabel(clf_path, survey_path, mode, name)
            trained_clf.autolabel()
            trained_clf.visual_check()
            
            # examine plot for a reasonable boundary between water and subaerial beach.
            
            trained_clf.apply()        
    """
    
    def __init__(self, clf_path, survey_path, mode, name = 'NAME_THIS_CLASSIFIER', **params):
        """
        Inputs
        ---------------
        clf_path: str
            Path to the classifier folder.
        survey_path: str or list
            Path(s) to surveys. Many surveys can be used at once to train a classifier, but only one
            survey can be automatically labelled at once. Visually checking the automatic labels 
            is important at this time. 
        mode: str, 'train' or 'label'
            Will you be training a classifier, or labeling the survey?
        name: str
            The name of the classifier folder. (default is 'NAME_THIS_CLASSIFIER')
        params: *optional*
            resolution: float
                The resolution of the survey to use for training or labeling.
                Needed for locating the right survey data.(default 0.5)
            classifier: str, 'gaussian process', 'mlp', or 'random forest'
                The type of classification algorithm to use. (default Gaussian Process)
            knn_clean: bool
                Whether or not to clean the automatically labelled survey using a nearest
                neighbor lookup. This will ensure that there are no mislabeled 'pockets' of
                water or unclassified data. (default False)
            k: int
                The number of neareset neighbors to use for knn cleanup.
                Ignored if knn_clean is False. (default 3)
            
        """
        
        self.clf_path = None
        
        if clf_path.endswith('/'):
            self.clf_path = clf_path
        else:
            self.clf_path = clf_path + '/'
        
        self.clf_path = clf_path
        self.survey_path = survey_path
        self.mode = mode
        self.name = name
        
            
        self.params = params
        
        self.parameters = None
        self.config_file()
        
        if self.mode == 'train':
            self.initialize_classifier()
        
        self.clf = None
        self.report = None
        self.labels = None
        
        
        
    def config_file(self):
        """This function builds the directory for the classifier and metadata.
        
        You likely will not need to use this function directly. It is run automatically when the
        class is instantiated.
        """
        
        path = self.clf_path

        clf_name = self.name
        #report = 'analysis_report'
        test_data = 'testing'
        train_data = 'training'

        clfnew = path + '/' + clf_name

        #report_folder = clfnew + '/' + report
        test_folder = clfnew + '/' + test_data
        train_folder = clfnew + '/' + train_data

        clf_folders = (clfnew, test_folder, train_folder)

        for elem in clf_folders:
            try:
                os.mkdir(elem)
            except:
                pass
        
        config_file = self.clf_path + '/' + self.name + '/config.yaml'
        
        if os.path.isfile(config_file):
            
            with open(config_file, 'r') as f:
                parameters = yaml.load(f)
                self.parameters = parameters
            
            if len(self.params):
                raise Exception('Cannot change parameters in existing configuration file. This ensures consistency among processed surveys in a folder. Create a new folder if you would like to experiment with new parameters.')
        else:
            rf_params = {
                "n_estimators": [80, 100, 150],
                "criterion": ['gini', 'entropy'],
                "max_depth": [None],
                "random_state": [0, 42, 97],
                "verbose": [0],
                "warm_start": [False],
                "class_weight": ['balanced', 'balanced_subsample']
            }

            gp_params = {
                "max_iter_predict" : [100,200],
                "random_state" : [0, 42, 97]
            }

            mlp_params = {
                "hidden_layer_sizes" : ((100,),(120,)),
                "activation" : ('logistic','relu'),
                "solver" : ('lbfgs','adam'),
                "learning_rate" : ('adaptive'),
                "max_iter" : [200,500],
                "random_state" : [0,42,97],
                "tol" : [1e-4, 5e-5],
                "verbose" : [False],
                "warm_start" : [True]
            }

            # TO DO: make classifier name ignore letter case.
            parameters = {
                # Random Forest, MLP, or Gaussian Process.
                "classifier": 'Gaussian Process',
                "resolution": 0.5,
                'clf_params': None,

                "knn_clean": False,
                "k": 3
            }

            if parameters["classifier"].lower() == 'gaussian process':
                parameters["clf_params"] = gp_params
            elif parameters["classifier"].lower() == 'mlp':
                parameters["clf_params"] = mlp_params
            elif parameters["classifier"].lower() == 'random forest':
                parameters["clf_params"] = rf_params
                
            parameters.update(self.params)
            
            self.parameters = parameters
            
            with open(config_file, 'w') as f:
                yaml.dump(parameters,f)
    
    
    def initialize_classifier(self):
        """This function creates the classifier defined by the configuration file.
        
        You will likely not need to use this directly. It is run automatically when the class
        is instantiated.
        """
        path = self.clf_path
        clf_name = self.name
        # report = 'analysis_report'
        test_data = 'testing'
        train_data = 'training'

        clfnew = path + '/' + clf_name
        
        if self.mode.lower() != 'train':
            raise Exception("mode must be 'train' in order to use function 'initialize_classifier'")
        else:
            pass
        
        
        params = self.parameters

        

        if params['classifier'].lower() == 'gaussian process':
            clfr = GaussianProcessClassifier()
            
        elif params['classifier'].lower() == 'random forest':
            clfr = RandomForestClassifier()
            
        elif params['classifier'].lower() == 'mlp':
            clfr = MLPClassifier()
            
        else:
            raise Exception(
                "Classifier type must be {'Gaussian Process', 'Random Forest', or 'MLP'}")
        
        clf = GridSearchCV(clfr, params['clf_params'],scoring = 'balanced_accuracy',verbose = 1)
        pipe = Pipeline([('scaler', StandardScaler()),
                             ('transformer', IncrementalPCA(n_components = 3, whiten = True)), ('learner', clf)])
        
        try:
            with open(clfnew  + '/init_clf.pickle','wb') as f:
                pickle.dump(pipe, f, pickle.HIGHEST_PROTOCOL)
        except:
            pass
        
        return pipe
    
    
    
    def train(self):
        """When in mode 'train', this function trains the classifier created by
        autolabel.initialize_classifier() using the surveys whose directories are given by the user 
        when the class is instantiated.
        """
        
        if self.mode.lower() != 'train':
            raise Exception("mode must be 'train' in order to use function 'train'")
        else:
            pass
        
        params = self.parameters
        
        clf_name = self.name
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

        clf_name = self.name
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
        clf_name = self.name
        clfnew = path + '/' + clf_name
        
        with open(clfnew  + '/trained_clf.pickle','wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        
        target_names = ['unclassified', 'water']
        
        self.report = classification_report(label_test, clf.predict(feat_test), output_dict = True,
                                            target_names = target_names)
        
        with open(clfnew + '/cv_report.yaml', 'w') as f:
            yaml.dump(self.report,f)
    

    def autolabel(self):
        """When in mode 'label', this function will label the survey given by the user when the
        class is instantiated.
        """
        
        if self.mode != 'label':
            raise Exception("Mode must be 'label' in order to use function 'autolabel'")
        else:
            pass
        
        if type(self.survey_path) == list:
            raise Exception("Survey Path must be a string with the path to a single survey to \
                            automatically label")
        else:
            pass
        
        params = self.parameters
        
        clf_name = self.name
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
        """This function allows the user to view the automatically labeled survey in order to verify
        that the labels look reasonable. 
        
        Inputs: any of the keyword arguments allowed by matplotlib.pyplot.scatter() will be accepted 
        by autolabel.visual_check().
        """
        
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
        
        fig0, ax0 = plt.subplots(1,1, figsize = [8,20])

        m = Basemap(projection='nsper', resolution='l', 
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
        """This function interpolates the predicted labels over the full resolution histogram from
        which texture features were extracted, then saves these labels into the survey folder. 
        """
        
        histpath = self.survey_path + '/resolution' + str(self.parameters["resolution"]) + '/1_histogram/'
        
        hist = np.load(histpath + 'hist.npz')
        
        X = np.vstack((self.labels[0],self.labels[1])).transpose()
        
        label_fix = self.labels[2]
        label_fix[label_fix == 1] == 0 # unclassified
        label_fix[label_fix == 2] == 9 # water
        
        labels = griddata(X, label_fix, (hist['x'],hist['y']), method = 'nearest')
        
        label_mask = ~np.isnan(hist['labels'])
        masked = labels * label_mask.astype(int)
        masked[masked ==0] = np.nan
        
        np.save(histpath + 'labels.npy', masked)
        
        
        
        
        
