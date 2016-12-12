import numpy as np

from projectionmodel import OrdinalModel, Experiment

class OrdinalModelExperiment(Experiment):
        
    def run(self):

        self.model = {'normalizer': OrdinalModel(self.data).fit(maxiter=10000, tolerance=0.01),}
                      #'projector': OrdinalModel(self.data, self.features, kerneldim=50).fit(maxiter=2000, tolerance=-np.inf)}

        return self
        
    def write_results(self, directory='params'):

        for m in self.model.values():
            m.write_params(directory)

if __name__ == '__main__':
    exp = OrdinalModelExperiment(data_dir='megaattitude_v1.csv')
    exp.run()