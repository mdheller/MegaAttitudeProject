import os
import numpy as np
import pandas as pd
import theano

from theano import tensor as T

from ..utility import *

class OrdinalModel(object):
    '''
    A cumulative link logit model of ordinal responses where each subject 
    is associated with their own freely varying set of cutpoints and each
    item (verb-frame pair) is associated with a real number
    '''

    gd_init = 100
    
    def __init__(self, data, features=None, kerneldim=0):
        '''
        Intitialize the ordinal model with observations in data.

        Params
        data (pandas.DataFrame): dataframe with columns for verb (str), frame (str), 
                                 subject (str), and response (int)
        '''
        
        self._create_ident()
        self._unpack_data(data, features)

        self.kerneldim = kerneldim
        self._initialized = False
        
    def _create_ident(self):
        '''Create a random identifier used to prevent name clashes on the GPU.'''
        
        self.ident = ''.join([str(i) for i in np.random.choice(9, size=10)])
        
    def _unpack_data(self, data, features):
        '''Associate variables in the data with instance attributes for less verbose access.'''

        if 'exclude' in data.columns:
            self.data = data[~data.exclude]
        else:
            self.data = data
            
        # shuffle data
        self.data = data.sample(frac=1)
        
        ## unpack acceptability data
        self._unpack_acceptability_data()

        ## unpack features
        self.syntactic_features = features
        
        if self.syntactic_features is not None:
            self._unpack_feature_data()
        
    def _unpack_acceptability_data(self):
        '''
        Convert subject, verb, and frame identifiers to categorical variables,
        map the levels of those categorical variables to indices, and wrap
        the indices in a theano shared variable, then extract the unique number
        of indices for each variable
        '''
        
        self.subj = self.data.participant.astype('category')
        self.verb = self.data.verb.astype('category')
        self.framevoice = self.data.framevoice.astype('category')

        self.subj_codes = theano.shared(np.array(self.subj.cat.codes),
                                        name=self.ident+'subjcodes')
        self.verb_codes = theano.shared(np.array(self.verb.cat.codes),
                                        name=self.ident+'verbcodes')
        self.frame_codes = theano.shared(np.array(self.framevoice.cat.codes),
                                         name=self.ident+'framecodes')
                
        self.response = np.array(self.data.response)

        ## count number of types
        self.num_of_subjects = self.subj.cat.categories.shape[0]
        self.num_of_verbs = self.verb.cat.categories.shape[0]
        self.num_of_frames = self.framevoice.cat.categories.shape[0]

    def _unpack_feature_data(self):
        '''
        Convert the featue matrix to a numpy array and extract the unique number
        of indices for each variable
        '''
        self.feature_names = self.syntactic_features.index
        self.features = np.array(self.syntactic_features)

        self.num_of_features = self.syntactic_features.shape[0]

        
    def _initialize_model(self, stochastic):
        '''
        Initialize the model by constructing latent variables, loss functions, 
        and updaters.
        '''
        
        self.representations = {}
        
        self._initialize_latent_variables()
        self._initialize_likelihood(stochastic)
        self._initialize_updaters(stochastic)

    def _initialize_latent_variables(self):
        '''Initialize the latent variables not associated with the response model.'''

        if self.syntactic_features is None:
            acceptability = np.random.normal(0., 1., size=[self.num_of_verbs, self.num_of_frames])

            self._acceptability = theano.shared(acceptability, name=self.ident+'acceptability')

            self.representations['acceptability'] = self._acceptability

        else:
            verb_features = np.random.normal(0., 1., size=[self.num_of_verbs, self.num_of_features])
            verb_features_t = theano.shared(verb_features, name=self.ident+'verb_features')

            self.representations['verb_features'] = verb_features_t
            
            self._verb_features = logisticT(verb_features_t)
             
            features_t = theano.shared(self.features, name=self.ident+'features')
    
            if self.kerneldim:
                noise_map = np.random.normal(0., 0.1, size=[self.kerneldim, self.num_of_features])
                
                if self.kerneldim < self.num_of_features:
                    kernelmap_aux = noise_map
                elif self.kerneldim == self.num_of_features:
                    kernelmap_aux = np.eye(self.num_of_features) + noise_map
                elif self.kerneldim < self.num_of_frames: # assumes that nframes > nfeatures
                    kernelmap_aux = np.linalg.pinv(2.*self.features-1.)[:self.kerneldim] + noise_map
                elif self.kerneldim == self.num_of_frames: # assumes that nframes > nfeatures
                    kernelmap_aux = np.linalg.pinv(2.*self.features-1.)# + noise_map
                    
                kernelmap_aux_t = theano.shared(kernelmap_aux, name=self.ident+'kernelmap')
                self._kernelmap = kernelmap_aux_t#logisticT(kernelmap_aux_t)

                vf_mapped = logisticT(T.dot(2.*self._verb_features-1., T.transpose(2.*self._kernelmap-1.)))
                # vf_mapped = logisticT(T.dot(self._verb_features, T.transpose(self._kernelmap)))
                # vf_mapped = T.dot(self._verb_features, T.transpose(self._kernelmap)) +\
                            #T.dot(1.-self._verb_features, 1.-T.transpose(self._kernelmap))

                sf_mapped = logisticT(T.dot(2.*self._kernelmap-1., 2.*features_t-1.))
                # sf_mapped = logisticT(T.dot(self._kernelmap, features_t))
                # vf_mapped = T.dot(self._kernelmap, features_t) +\
                              #T.dot(1.-self._kernelmap, 1.-features_t)

                acceptability = T.dot(2.*vf_mapped-1., 2.*sf_mapped-1.)
                # acceptability = T.dot(vf_mapped, sf_mapped) + T.dot(1.-vf_mapped, 1.-sf_mapped)
                
                # acceptability = T.dot(vf_mapped/self.kerneldim, sf_mapped/self.kerneldim) +\
                #                 T.dot(1.-vf_mapped/self.kerneldim, 1.-sf_mapped/self.kerneldim)

                self.representations['kernelmap'] = kernelmap_aux_t

                self._acceptability = logitT(acceptability / self.kerneldim)
            else:
                acceptability = T.dot(2.*self._verb_features-1., 2.*features_t-1.)
                # acceptability = T.dot(self._verb_features, features_t) +\
                #                 T.dot(1.-self._verb_features, 1.-features_t)

                self._acceptability = logitT(acceptability / self.num_of_features)
            
    def _initialize_likelihood(self, stochastic):
        '''Initialize the likelihood and loss functions.'''
        
        if stochastic:
            self._initialize_likelihood_stochastic()
        else:
            self._initialize_likelihood_batch()
            
        self._create_compute_likelihood()
        self._create_loss()
        self._create_information_criteria()

    def _initialize_likelihood_batch(self):
        '''Initialize the likelihood for batch gradient descent.'''
        
        cuts = self._initialize_cutpoints()
        
        response = theano.shared(self.response, name=self.ident+'response')

        response_max = np.max(self.response)
        response_min = np.min(self.response)
        
        upper = cuts[self.subj_codes, response] -\
                self._acceptability[self.verb_codes, self.frame_codes]

        lower = cuts[self.subj_codes, response-1] -\
                self._acceptability[self.verb_codes, self.frame_codes]

        probhigh = T.switch(T.lt(response, response_max),
                            logisticT(upper),
                            T.ones_like(response))

        problow = T.switch(T.gt(response, response_min),
                           logisticT(lower),
                           T.zeros_like(response))

        self.loglike = self.loglike_batch = T.sum(T.log(probhigh-problow+1e-20)) 

    def _initialize_likelihood_stochastic(self):
        '''Initialize the likelihood for stochastic gradient descent.'''

        raise NotImplementedError('Stochastic gradient descent is not currently implemented.')
        
        # cuts = self._initialize_cutpoints()
        
        ## TODO: this method should be replaced with a RandomStream-based method
        # order = theano.shared(np.random.choice(self.data.shape[0], size=1e6), name=self.ident+'order')

        # indices = order[T.arange((self.itr*stochastic)%int(1e6), ((self.itr+1)*stochastic)%int(1e6))]

        # upper = cuts[self.subj_codes[indices], response[indices]] -\
        #         self._acceptability[self.verb_codes[indices], self.frame_codes[indices]]

        # lower = cuts[self.subj_codes[indices], response[indices]-1] -\
        #         self._acceptability[self.verb_codes[indices], self.frame_codes[indices]]

        # probhigh = T.switch(T.lt(response[indices], response_max),
        #                     logisticT(upper),
        #                     1.)

        # problow = T.switch(T.gt(response[indices], response_min),
        #                    logisticT(lower),
        #                    0.)

        # self.loglike = T.sum(T.log(probhigh-problow+1e-20))
        # self.loglike_batch = self._create_loglike_batch()

    def _create_offset_and_jumps(self):
        offset_t = theano.shared(np.zeros(self.num_of_subjects), name=self.ident+'offset')
        
        jumps_aux = np.zeros([self.num_of_subjects, np.max(self.response)-1])
        jumps_aux = np.append(np.zeros([jumps_aux.shape[0],1])-np.inf, jumps_aux, axis=1)
        jumps_aux = np.append(jumps_aux, np.zeros([jumps_aux.shape[0],1])-np.inf, axis=1)
        
        jumps_aux_t = theano.shared(jumps_aux, name=self.ident+'jumps')

        return offset_t, jumps_aux_t
        
    def _initialize_cutpoints(self):
        '''Initialize the cutpoints for the cumulative link logit model.'''
        
        offset_t, jumps_aux_t = self._create_offset_and_jumps()
        
        jumps = T.exp(jumps_aux_t)

        self.representations['jumps'] = jumps_aux_t
        self.representations['offset'] = offset_t

        cutpoints_unshifted = T.extra_ops.cumsum(jumps, axis=1)

        response_max = np.max(self.response)
        midindex = int(response_max)/2 + 1

        return cutpoints_unshifted - cutpoints_unshifted[:,midindex][:,None] - offset_t[:,None]
        
    def _create_loglike_batch(self):
        '''
        Initialize the batch log-likelihood.
        
        Note: this exists so that stochastic optimizers can still 
              print the total log-likelihood
        '''
        
        probhigh_batch, problow_batch = self._initialize_likelihood_batch()

        return T.sum(T.log(probhigh_batch-problow_batch+1e-20))

    def _create_compute_likelihood(self):
        '''
        Initialize the python interface to the log-likelihood function.
        
        Note: this exists so that stochastic optimizers can still 
              print the total log-likelihood
        '''
        
        self._compute_likelihood = theano.function(inputs=[], outputs=self.loglike_batch)
        self.compute_likelihood = lambda: self._compute_likelihood()

    def _create_loss(self):
        '''
        Create the loss function.

        Note: this will get overridden in subclasses where the model is multiview
              or involves some prior
        '''
        
        self.loss = self.loglike

    def _create_information_criteria(self):
        '''Create the AIC and BIC functions.'''

        free_params = self._compute_free_params()

        self.aic = -2*(self.loglike_batch - free_params)
        self.bic = -2*self.loglike_batch + np.log(self.data.shape[0])*free_params

        self._compute_aic = theano.function(inputs=[], outputs=self.aic)
        self.compute_aic = lambda: self._compute_aic()        

        self._compute_bic = theano.function(inputs=[], outputs=self.bic)
        self.compute_bic = lambda: self._compute_bic()


    def _create_update_dicts(self, representations, stochastic):
        '''Create the dictionaries (as lists of tuples) that theano.update uses'''
        
        update_dict_gd = []
        update_dict_ada = []
        
        for name, rep in representations.items():
            ## compute gradient for current representation
            temp_grad = T.grad(self.loss, rep)

            ## clip gradient to avoid overflow
            rep_grad = T.switch(T.or_(rep > 10, rep < -10),
                                np.zeros(rep.shape.eval()), 
                                temp_grad)

            ## create a variable to store gradient history for adagrad
            rep_grad_hist_t = theano.shared(np.ones(rep.shape.eval()),
                                            name=self.ident+name+'_hist')

            ## create adagrad adjusted gradient
            rep_grad_adj = rep_grad / (T.sqrt(rep_grad_hist_t))

            if stochastic:
                gd_rate = ada_rate = 0.1
            else:
                gd_rate = 0.001
                ada_rate = .5 if name == 'jumps' else 1.
                ada_rate = .2 if name == 'offset' else ada_rate
                
            update_dict_gd += [(rep_grad_hist_t, rep_grad_hist_t + T.power(rep_grad, 2)),
                               (rep, rep + gd_rate*rep_grad)]
            update_dict_ada += [(rep_grad_hist_t, rep_grad_hist_t + T.power(rep_grad, 2)),
                                (rep, rep + ada_rate*rep_grad_adj)]  

        return update_dict_gd, update_dict_ada
        
    def _initialize_updaters(self, stochastic):
        '''Initialize the theano updater functions in a generic way.'''

        update_dict_gd, update_dict_ada = self._create_update_dicts(self.representations, stochastic)
        
        self._updater_gd = theano.function(inputs=[],
                                           outputs=[self.loss, self.loglike_batch, self.aic, self.bic],
                                           updates=update_dict_gd,
                                           name=self.ident+'updater_gd')

        self._updater_ada = theano.function(inputs=[],
                                            outputs=[self.loss, self.loglike_batch, self.aic, self.bic],
                                            updates=update_dict_ada,
                                            name=self.ident+'updater_ada')

    def _fit(self, updater_gd, updater_ada, maxiter, tolerance, verbose, stochastic):
        loss_stephist = []
        loglike_stephist = []

        last_loss = -1e20
        last_loglike = -1e20
        
        for i in np.arange(maxiter):
            
            if i < self.__class__.gd_init:
                loss, loglike, aic, bic = updater_gd()
            else:
                loss, loglike, aic, bic = updater_ada()

            loss_stephist.append(loss - last_loss)
            loglike_stephist.append(loglike-last_loglike)
            
            if verbose and not i % verbose:
                print i, loss, loglike, loglike_stephist[-1], aic, bic

            val = np.mean(loglike_stephist[-10:]) if i > 10 else np.mean(loglike_stephist)
            
            if val > tolerance or stochastic:
                last_loss = loss
                last_loglike = loglike

            else:
                if verbose:
                    print i, loss, loglike, loglike_stephist[-1], aic, bic
                break

        
    def fit(self, maxiter=15000, tolerance=0.01, stochastic=0, verbose=100):
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

        self._fit(self._updater_gd, self._updater_ada, maxiter, tolerance, verbose, stochastic)

        return self

    def compute_likelihood_aic_bic(self):
        '''Return -2*log-likelihood and the AIC'''
        
        ll = self.compute_likelihood()
        aic = self.compute_aic()
        bic = self.compute_bic()
        
        return ll, aic, bic

    def _compute_free_params(self):
        '''Return the number of free parameters in the model.'''

        return np.sum([np.prod(rep.shape.eval()) for rep in self.representations.values()])
    
    @property
    def acceptability(self):
        return pd.DataFrame(logisticT(self._acceptability).eval(),
                            index=self.verb.cat.categories,
                            columns=self.framevoice.cat.categories)

    @property
    def verb_features(self):
        if self.syntactic_features is None:
            raise ValueError, 'no features passed at initialization'
        else:
            return pd.DataFrame(self._verb_features.eval(),
                                index=self.verb.cat.categories,
                                columns=self.feature_names)
    
    @property
    def cutpoints(self):
        cutpoints = np.cumsum(np.exp(self.representations['jumps'].eval()[:,1:7]), axis=1)
        cutpoints -= cutpoints[:,int(np.max(self.response))/2][:,None]

        offset = self.representations['offset'].eval()

        cutpoints -= offset[:,None]
        
        cutpoints = pd.DataFrame(cutpoints, columns=['cutpoint'+str(i) for i in range(cutpoints.shape[1])])
        cutpoints['offset'] = offset
        
        return cutpoints

    def write_params(self):
        self.acceptability.to_csv(os.path.join(directory, 'acceptability_normalized.csv'))
        self.cutpoints.to_csv(os.path.join(directory, 'cutpoints.csv'))
