import sys, os, re, argparse, itertools
import numpy as np
import scipy as sp
import pandas as pd
import theano
import cPickle as pickle

from theano import tensor as T

from sklearn.decomposition import NMF

from .utility import *
from .ordinalmodel import OrdinalModel

class OrdinalFactorizer(OrdinalModel):
    '''
    A cumulative link logit model of ordinal responses where each subject 
    is associated with their own freely varying set of cutpoints and each
    item (verb-frame pair) is associated with a real number that is 
    deterministically related to a unit-valued row vector corresponding to 
    the verb and a unit-valued column vector corresponding to the frame by
    a fuzzy product logic disjunctive normal form
    '''

    gd_init = 10
    
    def __init__(self, data, features=None, num_of_typesigs=2, normalizer=None):
        '''
        Intitialize the ordinal factorizer with observations in data and hand-coded features.

        Params
        data     (pandas.DataFrame): dataframe with columns for verb (str), frame (str), 
                                     subject (str), and response (int)
        features (pandas.DataFrame): dataframe with columns for frame (str) and syntactic
                                     features of that frame

        '''

        self._create_ident()
        self._unpack_data(data, features)

        self.num_of_typesigs = num_of_typesigs
        self._normalizer = normalizer

        self._initialized = False
                                
    def _initialize_latent_variables(self):

        self._initialize_upper_latent_variables()
        self._initialize_acceptability()

    def _initialize_upper_latent_variables(self):
        if self._normalizer is None:
            if self.syntactic_features is None:
                project_aux = np.zeros([self.num_of_typesigs, self.num_of_frames])
            else:
                project_aux = np.zeros([self.num_of_typesigs, self.num_of_features])
                
            distmap_aux = np.zeros([self.num_of_verbs, self.num_of_typesigs])

        else:
            if self.syntactic_features is None:
                acc = np.array(self._normalizer.acceptability)
            else:
                acc = np.array(self._normalizer.verb_features)

            acc = logit(acc)-np.min(logit(acc))

            nmf = NMF(self.num_of_typesigs).fit(acc)
            
            proj = nmf.components_ + 0.1
            proj /= 1.1*np.max(proj)
            
            dist = nmf.transform(acc) + 0.1
            dist /= 1.1*np.max(dist)

            project_aux = logit(proj)
            distmap_aux = logit(dist)

        project_aux_t = theano.shared(project_aux, name=self.ident+'projection')
        distmap_aux_t = theano.shared(distmap_aux, name=self.ident+'distmap')

        self.representations = {'project' : project_aux_t,
                                'distmap' : distmap_aux_t}

        self.representations_pretrainable = {'project' : project_aux_t,
                                             'distmap' : distmap_aux_t}

        
        self._project = logisticT(project_aux_t)    
        self._distmap = logisticT(distmap_aux_t)

    def _initialize_acceptability(self):

        if self.syntactic_features is None:
            acceptability_tensor = self._distmap[:,:,None] * self._project[None,:,:]
            acceptability = 1.-T.prod(1.-acceptability_tensor, axis=1)
            self._acceptability = logitT(acceptability)

        else:
            verb_features_tensor = self._distmap[:,:,None] * self._project[None,:,:]
            verb_features = 1.-T.prod(1.-verb_features_tensor, axis=1)

            self._verb_features = logitT(verb_features)

            features_t = theano.shared(self.features, name=self.ident+'features')

            acceptability = T.dot(self._verb_features, features_t) +\
                            T.dot(1.-self._verb_features, 1.-features_t)
            
            self._acceptability = logitT(acceptability / self.num_of_features)

    def _initialize_cutpoints(self):
        
        if self._normalizer is None:
            offset_t = theano.shared(np.zeros(self.num_of_subjects), name=self.ident+'offset')
            
            jumps_aux = np.ones([self.num_of_subjects, np.max(self.response)-1])
            jumps_aux = np.append(np.zeros([jumps_aux.shape[0],1])-np.inf, jumps_aux, axis=1)
            jumps_aux = np.append(jumps_aux, np.zeros([jumps_aux.shape[0],1])-np.inf, axis=1)

        else:
            offset_t = theano.shared(self._normalizer.representations['offset'].eval(),
                                     name=self.ident+'offset')
            jumps_aux = self._normalizer.representations['jumps'].eval()

        jumps_aux_t = theano.shared(jumps_aux, name=self.ident+'jumps')
        jumps = T.exp(jumps_aux_t)

        self.representations['jumps'] = jumps_aux_t
        self.representations['offset'] = offset_t

        cutpoints_unshifted = T.extra_ops.cumsum(jumps, axis=1)

        response_max = np.max(self.response)
        midindex = int(response_max)/2
        
        return cutpoints_unshifted - cutpoints_unshifted[:,midindex][:,None] - offset_t[:,None]

    def _initialize_updaters(self, stochastic):
        '''Initialize the theano updater functions in a generic way.'''

        OrdinalModel._initialize_updaters(self, stochastic)
        
        update_dict_gd, update_dict_ada = self._create_update_dicts(self.representations_pretrainable, stochastic)
        
        self._updater_gd_pretrain = theano.function(inputs=[],
                                                    outputs=[self.loss, self.loglike_batch, self.aic, self.bic],
                                                    updates=update_dict_gd,
                                                    name=self.ident+'updater_gd_pretrain')

        self._updater_ada_pretrain = theano.function(inputs=[],
                                                     outputs=[self.loss, self.loglike_batch, self.aic, self.bic],
                                                     updates=update_dict_ada,
                                                     name=self.ident+'updater_ada_pretrain')

        
    def fit(self, pretrainiter=10, maxiter=15000, tolerance=0.01, stochastic=0, verbose=100):
        '''
        Fit the model.

        Params
        maxiter      (int): maximum number of iterations if tolerance is not reached
        tolerance  (float): stopping threshold for average step-size over 10 iterations
        stochastic   (int): number of training instances per update in stochastic gradient descent
                            set to 0 for batch gradient descent
        verbose      (int): number of iterations between printing each update
                            set to 0 for no verbosity
        '''
        
        if not self._initialized:
            self._initialize_model(stochastic)
            self._initialized = True

        self._fit(self._updater_gd_pretrain, self._updater_ada_pretrain, pretrainiter, tolerance, verbose, stochastic)
        self._fit(self._updater_gd, self._updater_ada, maxiter, tolerance, verbose, stochastic)

        return self

        
    @property
    def projection(self):
        return pd.DataFrame(self._project.eval(),
                            index=range(self.num_of_typesigs),
                            columns=self.framevoice.cat.categories)

    @property
    def verbtypesig(self):
        return pd.DataFrame(self._distmap.eval(),
                            index=self.verb.cat.categories,
                            columns=range(self.num_of_typesigs))
    
    def write_params(self, directory):
        self.acceptability.to_csv(os.path.join(directory, 'acceptability'+str(self.num_of_typesigs)+'.csv'))
        self.projection.to_csv(os.path.join(directory, 'projection'+str(self.num_of_typesigs)+'.csv'))
        self.verbtype.to_csv(os.path.join(directory, 'verbtype'+str(self.num_of_typesigs)+'.csv'))
