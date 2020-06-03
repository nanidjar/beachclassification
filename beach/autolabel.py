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
            raise Exception("mode must be 'label' in order to use function 'autolabel'")
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
                new_labels[i] = nlab[0]
        else:
            labels = dirty_labels
        
        self.labels = labels
        
        
        
        
    def apply(self):
        
       # TO DO: 'apply' will create a 2D mesh the same size and resolution as the survey histogram, filled with labels corresponding to the nearest neighbor predicted label given at spacing 'step_size' given in self.labels. The indices of existing data in hist will provide a mask for this label mesh. pixels without data will be given a label value of nan. This final label mesh will be saved in the same directory as 'hist.npz' of the survey with the name 'labels.npy'
    
        pass