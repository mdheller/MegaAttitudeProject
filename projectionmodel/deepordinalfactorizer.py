import sys, os, re, argparse, itertools
import numpy as np
import scipy as sp
import pandas as pd
import theano
import cPickle as pickle

from theano import tensor as T

from sklearn.decomposition import NMF

from .utility import *
from .ordinalfactorizer import OrdinalFactorizer

class DeepOrdinalFactorizer(OrdinalFactorizer):
    gd_init = 10000
    
    def __init__(self, data, features=None, num_of_typesigs=2, num_of_types=2, precision=np.inf, normalizer=None):
        OrdinalFactorizer.__init__(self, data, features, num_of_typesigs, normalizer)

        self.num_of_types = num_of_types
        self.precision = precision
        
        if self._normalizer is not None:
            self._check_normalizer()
        
    def _check_normalizer(self):
        
        try:
            assert isinstance(self._normalizer, OrdinalFactorizer)
        except AssertionError:
            raise ValueError('normalizer must be an OrdinalFactorizer')

        try:
            assert self._normalizer.num_of_typesigs == self.num_of_typesigs
        except AssertionError:
            raise ValueError('num_of_typesigs must match normalizer\'s num_of_typesigs')
        
    def _initialize_upper_latent_variables(self):
        if self._normalizer is None:
            typerep_aux = np.zeros([self.num_of_verbs, self.num_of_types])
            typemap_aux = np.zeros([self.num_of_types, self.num_of_typesigs])
            projmap_aux = np.zeros([self.num_of_frames, self.num_of_types])
            
        else:                
            vt = np.array(self._normalizer.verbtypesig)
            vt = logit(vt)-np.min(logit(vt))

            
            nmf = NMF(self.num_of_types).fit(vt)

            trep = nmf.transform(vt) + 0.1
            trep /= 1.1*np.max(trep)
            
            tmap = nmf.components_ + 0.1
            tmap /= 1.1*np.max(tmap)
            
            typerep_aux = logit(trep)
            typemap_aux = logit(tmap)
            projmap_aux = np.zeros([self.num_of_frames, self.num_of_types])

        typerep_aux_t = theano.shared(typerep_aux, name=self.ident+'typerep')
        typemap_aux_t = theano.shared(typemap_aux, name=self.ident+'typemap')
        projmap_aux_t = theano.shared(projmap_aux, name=self.ident+'projmap')
        
        self.representations = {'typerep' : typerep_aux_t,
                                'typemap' : typemap_aux_t,
                                'projmap' : projmap_aux_t}

        self.representations_pretrainable = {'typerep' : typerep_aux_t,
                                             'typemap' : typemap_aux_t,
                                             'projmap' : projmap_aux_t}

        
        indices = range(self.num_of_types)

        typerep = logisticT(typerep_aux_t)
        typemap = logisticT(typemap_aux_t)
        projmap = logisticT(projmap_aux_t)

        distmap_tensor = typerep[:,:,None] * typemap[None,:,:]
        self._distmap_predicted = 1.-T.prod(1.-distmap_tensor, axis=1)

        project_tensor = projmap[:,:,None] * typemap[None,:,:]
        self._project_predicted = T.transpose(1.-T.prod(1.-project_tensor, axis=1))

        if self.precision < np.inf:
            if self._normalizer is None:
                project_aux = np.zeros([self.num_of_typesigs, self.num_of_frames])
                distmap_aux = np.zeros([self.num_of_verbs, self.num_of_typesigs])

            else:
                project_aux = logit(np.array(self._normalizer.projection))
                distmap_aux = logit(np.array(self._normalizer.verbtypesig))

            project_aux_t = theano.shared(project_aux, name=self.ident+'project')
            distmap_aux_t = theano.shared(distmap_aux, name=self.ident+'distmap')

            self.representations['project'] = project_aux_t
            self.representations['distmap'] = distmap_aux_t

            self._project = project = logisticT(project_aux_t)
            self._distmap = distmap = logisticT(distmap_aux_t)
            
        else:
            self._project = self._project_predicted
            self._distmap = self._distmap_predicted

    def _initialize_verbtype_projection_likelihood(self):
        alpha_distmap = self.precision*self._distmap_predicted
        beta_distmap = self.precision*(1.-self._distmap_predicted)

        prior_distmap = T.sum(betapriorln(self._distmap, alpha_distmap, beta_distmap))
        
        alpha_project = self.precision*self._project_predicted
        beta_project = self.precision*(1.-self._project_predicted)

        prior_project = T.sum(betapriorln(self._project, alpha_project, beta_project))
        
        return prior_distmap + prior_project
            
    def _initialize_likelihood_batch(self):
        OrdinalFactorizer._initialize_likelihood_batch(self)

        if self.precision < np.inf:
            self.loglike += self._initialize_verbtype_projection_likelihood()

    def _initialize_likelihood_stochastic(self):
        raise NotImplementedError, 'SGD with finite precision not implemented'
    
        OrdinalFactorizer._initialize_likelihood_stochastic(self)

        if self.precision < np.inf:
            self.loglike += self._initialize_verbtype_projection_likelihood()
            self.loglike_batch += self._initialize_verbtype_projection_likelihood()
        
    @property
    def frametype(self):
        return pd.DataFrame(logisticT(self.representations['projmap']).eval(),
                            index=self.framevoice.cat.categories,
                            columns=range(self.num_of_types))

    @property
    def verbtype(self):
        return pd.DataFrame(logisticT(self.representations['typerep']).eval(),
                            index=self.verb.cat.categories,
                            columns=range(self.num_of_types))

    @property
    def typetypesig(self):
        return pd.DataFrame(logisticT(self.representations['typemap']).eval(),
                            index=range(self.num_of_types),
                            columns=range(self.num_of_typesigs))
