import numpy as np

from projectionmodel import OrdinalModel, OrdinalFactorizer, DeepOrdinalFactorizer, Experiment

class DeepOrdinalFactorizerExperiment(Experiment):
        
    def run(self, max_num_of_typesigs=2, max_num_of_types=2, precision=10.):

        self.factorizer = {}
        self.model = {}

        normalizer = OrdinalModel(self.data).fit(maxiter=10000, tolerance=10.)

        self.factorizer[12] = OrdinalFactorizer(self.data, #self.features,
                                               num_of_typesigs=12,
                                               normalizer=normalizer).fit(maxiter=10000, tolerance=1.)


        for i in range(8, 9):
            print '\n', i, '\n'
            self.model[i] = DeepOrdinalFactorizer(self.data, #self.features,
                                                  num_of_typesigs=12,
                                                  num_of_types=i,
                                                  precision=precision,
                                                  normalizer=self.factorizer[12]).fit(pretrainiter=10,
                                                                                      maxiter=10000,
                                                                                      tolerance=-np.inf)
        
        # for i in range(2, max_num_of_typesigs+1):
        #     self.model[i] = OrdinalFactorizer(self.data, #self.features,
        #                                       num_of_typesigs=i).fit(maxiter=1000, tolerance=0.01)

        return self
            
    def write_results(self, directory='params'):

        for m in self.model.values():
            m.write_params(directory)
        
if __name__ == '__main__':
    exp = DeepOrdinalFactorizerExperiment().run()
