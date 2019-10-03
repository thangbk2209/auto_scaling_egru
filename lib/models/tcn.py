from config import *


class TcnModel:

    def __init__(self):
        self.tcn_config = Config.TCN_CONFIG
    
    def __fit_with_ga(self):
        print('>>> Fit with genetic algorithm <<<')
    
    def __fit_with_pso(self):
        print('>>> Fit with particle swarm optimization algorithm <<<')
    
    def __fit_with_bp(self):
        print('>>> Fit with back propagation algorithm <<<')

    def __fit_with_sfo(self):
        print('>>> Fit with sf optimization algorithm <<<')
    
    def __fit_with_isfo(self):
        print('>>> Fit with isf optimization algorithm <<<')

    def fit(self):
        if Config.OPTIMIZATION_METHOD == 'ga':
            self.__fit_with_ga()
        elif Config.OPTIMIZATION_METHOD == 'pso':
            self.__fit_with_pso()
        elif Config.OPTIMIZATION_METHOD == 'bp':
            self.__fit_with_bp()
        elif Config.OPTIMIZATION_METHOD == 'sfo':
            self.__fit_with_sfo()
        elif Config.OPTIMIZATION_METHOD == 'isfo':
            self.__fit_with_isfo()
        else:
            print('[ERROR] Please check your optimization method! We do not support {}'.format(Config.OPTIMIZATION_METHOD))

