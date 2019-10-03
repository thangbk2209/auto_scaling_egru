import numpy as np

from config import *


class DataPreprocessor:

    def __init__(self, data=None, train_size=0.6, valid_size=0.2):
        self.data = data
        self.train_size = train_size
        self.valid_size = valid_size
    
    def create_timeseries(self, X):
        if len(X) > 1:
            data = np.concatenate((X[0], X[1]), axis=1)
            if len(X) > 2:
                for i in range(2, len(X), 1):
                    data = np.column_stack((data, X[i]))
        else:
            data = []
            for i in range(len(X[0])):
                data.append(X[0][i])
            data = np.array(data)
        return data

    def create_x(self, timeseries, sliding):
        data_x = []
        for i in range(len(timeseries) - sliding):
            _data = []
            for j in range(sliding):
                _data.append(timeseries[i + j])
            data_x.append(_data)
        return data_x

    def init_data_rnn(self, sliding):
        print('>>> start init data for training LSTM model <<<')

        if Config.DATA_EXPERIMENT == 'traffic':
            x_data = [self.data]
            y_data = self.data
        else:
            print('    [ERROR] We do not support your data for preprocessing!')

        x_timeseries = self.create_timeseries(x_data)
        num_points = x_timeseries.shape[0]
        train_point = int(self.train_size * num_points)
        valid_point = int((self.train_size + self.valid_size) * num_points)

        x_sampple = self.create_x(x_timeseries, sliding)

        x_train = x_sampple[0:train_point - sliding]
        x_train = np.array(x_train)

        x_valid = x_sampple[train_point - sliding: valid_point - sliding]
        x_valid = np.array(x_valid)

        x_test = x_sampple[valid_point - sliding:]
        x_test = np.array(x_test)

        y_train = y_data[sliding: train_point]
        y_train = np.array(y_train)

        y_valid = y_data[train_point: valid_point]
        y_valid = np.array(y_valid)

        y_test = y_data[valid_point:]
        y_test = np.array(y_test)

        print('>>> Init data for training LSTM model done <<<')
        return x_train, y_train, x_valid, y_valid, x_test, y_test