import multiprocessing
from multiprocessing import Pool
from queue import Queue
from sklearn.model_selection import ParameterGrid, train_test_split

from lib.models.rnn import RnnModel
from lib.models.tcn import TcnModel
from config import *


class ModelTrainer:

    def __init__(self, data, scaler):
        self.data = data
        self.scaler = scaler
        self.rnn_config = Config.RNN_CONFIG
        self.bp_config = Config.BP_CONFIG
        self.ga_config = Config.GA_CONFIG
        self.pso_config = Config.PSO_CONFIG
        self.sfo_config = Config.SFO_CONFIG
        self.input_config = Config.INPUT_CONFIG

    def __fit_with_rnn(self, item):
        sliding = item['sliding']
        train_size = item['train_size']
        valid_size = item['valid_size']
        activation = item['activation']
        num_units = item['num_units']
        rnn_model = RnnModel(
            data=self.data, scaler=self.scaler, sliding=sliding, train_size=train_size, valid_size=valid_size,
            activation=activation, num_units=num_units
            )
        if Config.OPTIMIZATION_METHOD == 'bp':
            optimizers = item['optimizers']
            batch_size = item['batch_size']
            rnn_model.fit_with_bp(optimizers, batch_size)
        else:
            num_particle = item['num_particles']
            if Config.OPTIMIZATION_METHOD == 'ga':
                rnn_model.fit_with_ga(num_particle)
            elif Config.OPTIMIZATION_METHOD == 'pso':
                rnn_model.fit_with_pso(num_particle)
            elif Config.OPTIMIZATION_METHOD == 'sfo':
                rnn_model.fit_with_sfo(num_particle)
            else:
                error_notification = '[ERROR] Please check your optimization method! We do not support {}'\
                    .format(Config.OPTIMIZATION_METHOD)
                print(error_notification)

    def __train_with_rnn(self):
        print('  [2] >>> Training with rnn model type <<<')
        if Config.OPTIMIZATION_METHOD == 'ga':
            param_grid = {
                'num_particles': self.ga_config['num_particles'],
                'sliding': self.input_config['sliding'],
                'train_size': self.input_config['train_size'],
                'valid_size': self.input_config['valid_size'],
                'activation': self.rnn_config['activation'],
                'num_units': self.rnn_config['num_units']
            }
        elif Config.OPTIMIZATION_METHOD == 'pso':
            param_grid = {
                'num_particles': self.pso_config['num_particles'],
                'sliding': self.input_config['sliding'],
                'train_size': self.input_config['train_size'],
                'valid_size': self.input_config['valid_size'],
                'activation': self.rnn_config['activation'],
                'num_units': self.rnn_config['num_units']
            }
        elif Config.OPTIMIZATION_METHOD == 'sfo':
            param_grid = {
                'num_particles': self.sfo_config['num_particles'],
                'sliding': self.input_config['sliding'],
                'train_size': self.input_config['train_size'],
                'valid_size': self.input_config['valid_size'],
                'activation': self.rnn_config['activation'],
                'num_units': self.rnn_config['num_units']
            }
        elif Config.OPTIMIZATION_METHOD == 'bp':
            param_grid = {
                'sliding': self.input_config['sliding'],
                'train_size': self.input_config['train_size'],
                'valid_size': self.input_config['valid_size'],
                'activation': self.rnn_config['activation'],
                'num_units': self.rnn_config['num_units'],
                'optimizers': self.bp_config['optimizers'],
                'batch_size': self.bp_config['batch_size']
            }
        queue = Queue()
        for item in list(ParameterGrid(param_grid)):
            # queue.put_nowait(item)
            self.__fit_with_rnn(item)
        # pool = Pool(Config.NUM_THREAD)
        # pool.map(self.__fit_with_rnn, list(queue.queue))
        # pool.close()
        # pool.join()
        # pool.terminate()
        print('  [2] >>> Training with lstm model complete <<<')

    def __train_with_tcn(self):
        print('  [2] >>> Training with tcn model <<<')
        tcn_model = TcnModel()
        tcn_model.fit()
        print('  [2] >>> Training with tcn model complete <<<')

    def train(self):
        if Config.MODEL_TYPE in ['gru', 'lstm', 'rnn', 'agru', 'alstm','arnn']:
            self.__train_with_rnn()
        elif Config.MODEL_TYPE == 'tcn':
            self.__train_with_tcn()
        else:
            print('[ERROR]: Please check your model type. We do not suport {}'.format(Config.MODEL_TYPE))
