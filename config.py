import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')

class Config:
    PLT_OPTION = 'TkAgg'
    MODEL_TYPE = 'alstm'  # gru, lstm, rnn, alstm, agru, tcn
    OPTIMIZATION_METHOD = 'bp'  # ga, pso, sfo, isfo, bp
    DATA_EXPERIMENT = 'traffic'

    TRAFFIC_DATA_CONFIG = {
        'data_location': 'eu',  # eu, uk
        'file_data_name': '/input_data/traffic/it_{}_5m.csv',
        'data_path': CORE_DATA_DIR + '{}',
        'colnames': ['timestamp', 'bit', 'byte', 'kilobyte', 'megabyte']
    }

    # Information of train, validation and test set
    TRAIN_SIZE = 0.6
    VALID_SIZE = 0.2
    
    # number iteration for training
    EPOCHS = 200

    INPUT_CONFIG = {
        'sliding': [5],
        'train_size': [0.6],
        'valid_size': [0.2]
    }

    RNN_CONFIG = {
        'num_units': [[2]],
        'activation': ['tanh']  # 'sigmoid', 'relu', 'tanh', 'elu'
    }

    TCN_CONFIG = {
        'activation': ['tanh']
    }

    BP_CONFIG = {
        'batch_size': [8],
        'optimizers': ['adam'],  # 'momentum', 'adam', 'rmsprop'  
    }

    GA_CONFIG = {
        'num_particles': [100]
    }

    PSO_CONFIG = {
        'num_particles': [100]
    }

    SFO_CONFIG = {
        'num_particles': [100]
    }

    NUM_THREAD = 1
