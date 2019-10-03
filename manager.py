from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

from config import *
from lib.model_training import ModelTrainer


def read_data():
    traffic_data_config = Config.TRAFFIC_DATA_CONFIG
    
    if Config.DATA_EXPERIMENT == 'traffic':
        file_data_name = traffic_data_config['file_data_name'].format(traffic_data_config['data_location'])
        normal_data_file = PROJECT_DIR + '/data/input_data/{}/normalized_data.pkl'.format(Config.DATA_EXPERIMENT)
        colnames = traffic_data_config['colnames']
        file_data_path = traffic_data_config['data_path'].format(file_data_name)

        df = read_csv(file_data_path, header=None, index_col=False, names=colnames, engine='python')
        traffic = df['megabyte'].values.reshape(-1, 1)

        traffic_scaler = MinMaxScaler(feature_range=(0, 1))
        traffic_normal = traffic_scaler.fit_transform(traffic)

        return traffic_normal, traffic_scaler
    else:
        print('>>> We do not support to experiment with this data <<<')
        return None, None


def init_model():
    print('[1] >>> Start init model')
    normalized_data, scaler = read_data()
    model_trainer = ModelTrainer(normalized_data, scaler)
    model_trainer.train()
    print('[1] >>> Init model complete')


if __name__ == "__main__":
    init_model()
