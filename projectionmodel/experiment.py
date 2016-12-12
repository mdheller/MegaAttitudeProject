import re
import numpy as np
import pandas as pd
import theano

class Experiment(object):

    def __init__(self, data_dir='../../MegaAttitude/data/full/full.filtered.csv',
                 features_dir='features_asw_reduced.csv',
                 randomseed=5457):

        np.random.seed(randomseed)

        self._load_data(data_dir)
        self._load_features(features_dir)
        
    def _load_data(self, data_dir):
        ## load data
        data = pd.read_csv(data_dir, low_memory=False)

        ## add column that combines frame and voice
        data['framevoice'] = data.frame + ':' + data.voice

        ## remove nan responses
        data = data[~data.response.isnull()]
        data.response = data.response.astype(int)
        
        ## add data as an attribute of the experiment
        self.data = data
    
    def _load_features(self, features_dir):
        ## create verb-by-frame+voice matrix
        verbframe = self.data.pivot_table(index=['verb'], columns=['framevoice'],
                                          aggfunc=lambda x: np.mean(x), values='response')

        ## load features
        framevoice = pd.DataFrame({'framevoice': verbframe.columns,
                                   'frame' : verbframe.columns.map(lambda x: x.split(':')[0]),
                                   'voice' : verbframe.columns.map(lambda x: x.split(':')[1])})

        features = pd.merge(pd.read_csv(features_dir), framevoice).drop(['frame'], axis=1)
        features.index = features['framevoice'].map(lambda x: re.sub('_', ' ', x))
        features = features.drop(['framevoice'], axis=1)#.ix[verbframe.columns]
        features.columns = features.columns.map(lambda x: re.sub('_', ' ', x))
        features = pd.get_dummies(features).transpose().astype(theano.config.floatX)
        features = features.drop([col for col in features.index if col.split('_')[-1] in ['NONE', 'active']])

        features.ix['npanim_bare'] = features.ix['voice_passive'] + features.ix['npanim_bare']

        features.index = features.index.map(lambda x: re.sub('_', ':', x)) 

        self.features = features
        
    def run(self, max_num_of_typesigs=15, max_num_of_types=0):
        raise NotImplementedError
        
    def write_results(self, directory='params'):
        raise NotImplementedError
