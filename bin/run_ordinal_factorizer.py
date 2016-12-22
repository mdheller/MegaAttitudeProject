from megaattitude import OrdinalModel, OrdinalFactorizer, Experiment

class OrdinalFactorizerExperiment(Experiment):
        
    def run(self, max_num_of_typesigs=2):

        self.model = {}

        normalizer = OrdinalModel(self.data).fit(maxiter=10000, tolerance=10.)
        
        self.model[4] = OrdinalFactorizer(self.data, #self.features,
                                          num_of_typesigs=4,
                                          normalizer=normalizer).fit(maxiter=1000, tolerance=0.01)
        
        # for i in range(2, max_num_of_typesigs+1):
        #     self.model[i] = OrdinalFactorizer(self.data, #self.features,
        #                                       num_of_typesigs=i).fit(maxiter=1000, tolerance=0.01)

        return self
            
    def write_results(self, directory='params'):

        for m in self.model.values():
            m.write_params(directory)
        
if __name__ == '__main__':
    exp = OrdinalFactorizerExperiment(data_dir='data/megaattitude_v1.csv',
                                      features_dir='data/features_asw_reduced.csv',
                                      randomseed=5457)
    exp.run()
