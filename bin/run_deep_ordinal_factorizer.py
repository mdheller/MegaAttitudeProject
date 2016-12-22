import numpy as np

from megaattitude import OrdinalModel, OrdinalFactorizer, DeepOrdinalFactorizer, Experiment

class DeepOrdinalFactorizerExperiment(Experiment):
        
    def run(self, max_num_of_typesigs=2, max_num_of_types=2, precision=10.):

        self.factorizer = {}
        self.model = {}

        normalizer = OrdinalModel(self.data).fit(maxiter=10000, tolerance=10.)

        for i in range(2, max_num_of_typesigs+1):
            print '\n', i, '\n'
            
            self.factorizer[i] = OrdinalFactorizer(self.data, #self.features,
                                       num_of_typesigs=12,
                                       normalizer=normalizer).fit(maxiter=10000, tolerance=1.)

            for j in range(2, max_num_of_types+1):
                self.model[j] = DeepOrdinalFactorizer(self.data, #self.features,
                                                      num_of_typesigs=i,
                                                      num_of_types=j,
                                                      precision=precision,
                                                      normalizer=self.factorizer[i]).fit(pretrainiter=10,
                                                                                         maxiter=10000,
                                                                                         tolerance=-np.inf)

        return self
            
    def write_results(self, directory='params'):

        for m in self.model.values():
            m.write_params(directory)
        
if __name__ == '__main__':
    exp = DeepOrdinalFactorizerExperiment(data_dir='data/megaattitude_v1.csv',
                                          features_dir='data/features_asw_reduced.csv',
                                          randomseed=5457)
    exp.run()
