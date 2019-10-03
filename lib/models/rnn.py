import tensorflow as tf
import keras
from keras import optimizers, regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import LSTM, GRU, RNN, Activation, Dense, Dropout, Input
from keras.models import Model, Sequential, model_from_json
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as r2
import matplotlib

from config import *
from lib.data_preprocessing import DataPreprocessor
from lib.evolutionary_algorithms.pso import *
from lib.evolutionary_algorithms.ga import *
from lib.evolutionary_algorithms.sfo import *
from lib.evolutionary_algorithms.woa import *
matplotlib.use(Config.PLT_OPTION)
import matplotlib.pyplot as plt


class RnnModel:

    def __init__(self, data=None, scaler=None, sliding=None, train_size=None, valid_size=None,
                 activation=None, num_units=None):
        self.data = data
        self.scaler = scaler
        self.sliding = sliding
        self.train_size = train_size
        self.valid_size = valid_size
        self.activation = activation
        self.num_units = num_units

        self.preprocessing_data()

    def preprocessing_data(self):
        data_preprocessor = DataPreprocessor(self.data, self.train_size, self.valid_size)
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = \
            data_preprocessor.init_data_rnn(sliding=self.sliding)
        print('========== Sample information ===========')
        print('self.x_train.shape = {}'.format(self.x_train.shape))
        print('self.y_train.shape = {}'.format(self.y_train.shape))
        print('self.x_valid.shape = {}'.format(self.x_valid.shape))
        print('self.y_valid.shape = {}'.format(self.y_valid.shape))
        print('self.x_test.shape = {}'.format(self.x_test.shape))
        print('self.y_test.shape = {}'.format(self.y_test.shape))
        print('=========================================')

    def init_RNN(self):
        num_layers = len(self.num_units)
        hidden_layers = []
        for i in range(num_layers):
            # if i == 0:
            if Config.MODEL_TYPE == 'lstm' or Config.MODEL_TYPE == 'alstm':
                cell = tf.nn.rnn_cell.LSTMCell(self.num_units[i], activation=self.activation)
            elif Config.MODEL_TYPE == 'gru' or Config.MODEL_TYPE == 'agru':
                cell = tf.nn.rnn_cell.GRUCell(self.num_units[i], activation=self.activation)
            elif Config.MODEL_TYPE == 'rnn' or Config.MODEL_TYPE == 'arnn':
                cell = tf.nn.rnn_cell.RNNCell(self.num_units[i], activation=self.activation)
                # if self.variation_dropout:
                #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                #                                          state_keep_prob=self.dropout_rate, variational_recurrent=True,
                #                                          input_size=self.x_train.shape[2], dtype=tf.float32)
            else:
                print('[ERROR] Please check your model type. We do not support {}!!!'.format(Config.MODEL_TYPE))
            hidden_layers.append(cell)
            # else:
                # cell = tf.nn.rnn_cell.LSTMCell(num_units[i], activation=activation)
                # if self.variation_dropout:
                #     cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=self.dropout_rate,
                #                                          state_keep_prob=self.dropout_rate, variational_recurrent=True,
                #                                          input_size=num_units[i - 1], dtype=tf.float32)
                # hidden_layers.append(cell)
        rnn_cells = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple=True)
        return rnn_cells

    def mlp(self, input, num_units, activation):
        num_layers = len(num_units)
        prev_layer = input
        for i in range(num_layers):
            prev_layer = tf.layers.dense(prev_layer, num_units[i], activation=activation, name='layer' + str(i))

        prediction = tf.layers.dense(inputs=prev_layer, units=1, activation=activation, name='output_layer')
        return prediction

    def early_stopping_desition(self, array, patience):
        value = array[len(array) - patience - 1]
        arr = array[len(array) - patience:]
        check = 0
        for val in arr:
            if(val > value):
                check += 1
        if check == patience:
            return False
        else:
            return True

    def draw_train_loss(self, cost_train_set, cost_valid_set):
        plt.plot(cost_train_set)
        plt.plot(cost_valid_set)

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(self.train_loss_path + self.file_name + '.png')
        plt.close()

    def compute_error(self, y_pred, y_true):
        MAE_err = MAE(y_pred, y_true)
        RMSE_err = np.sqrt(MSE(y_pred, y_true))
        r2_err = r2(y_pred, y_true)
        return MAE_err, RMSE_err, r2_err

    def _build_rnn_model(self):
        tf.reset_default_graph()
        x = tf.placeholder("float", [None, self.sliding, self.x_train.shape[2]], name='x')
        y = tf.placeholder("float", [None, self.y_train.shape[1]], name='y')
        with tf.variable_scope('LSTM'):
            lstm_layer = self.init_RNN()
            outputs, new_state = tf.nn.dynamic_rnn(lstm_layer, x, dtype="float32")
            outputs = tf.identity(outputs, name='outputs_lstm')

        prediction = tf.layers.dense(outputs[:, :, -1], self.y_train.shape[1], activation=self.activation, 
                                     use_bias=True)
        prediction = tf.identity(prediction, name='prediction')
        # Loss is mean square error
        loss = tf.reduce_mean(tf.square(y - prediction))
        loss = tf.identity(loss, name='loss')
        # optimize = optimizer.minimize(loss)

        graph = tf.get_default_graph()
        return x, y, outputs, new_state, prediction, loss, graph

    def _build_arnn_model(self):
        tf.reset_default_graph()
        x = tf.placeholder("float", [None, self.sliding, self.x_train.shape[2]], name='x')
        y = tf.placeholder("float", [None, self.y_train.shape[1]], name='y')
        with tf.variable_scope('LSTM'):
            lstm_layer = self.init_RNN()
            outputs, new_state = tf.nn.dynamic_rnn(lstm_layer, x, dtype="float32")
            outputs = tf.identity(outputs, name='outputs_lstm')
        attention_weights = tf.nn.softmax(tf.layers.dense(outputs, 1, activation=self.activation), axis=1)
        # prediction = tf.layers.dense(outputs[:, :, -1], self.y_train.shape[1], activation=self.activation, 
        #                                   use_bias=True)
        context_vector = attention_weights * outputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        prediction = tf.layers.dense(context_vector, self.y_train.shape[1], activation=self.activation, 
                                     use_bias=True)
        prediction = tf.identity(prediction, name='prediction')
        # Loss is mean square error
        loss = tf.reduce_mean(tf.square(y - prediction))
        loss = tf.identity(loss, name='loss')
        # optimize = optimizer.minimize(loss)

        graph = tf.get_default_graph()
        return x, y, outputs, new_state, prediction, loss, graph

    def build_model(self):
        if Config.MODEL_TYPE == 'lstm' or Config.MODEL_TYPE == 'gru' or Config.MODEL_TYPE == 'rnn':
            return self._build_rnn_model()
        elif Config.MODEL_TYPE == 'alstm' or Config.MODEL_TYPE == 'agru' or Config.MODEL_TYPE == 'arnn':
            return self._build_arnn_model()

    def fit_with_ga(self, num_particle):
        print('    [3] >>> Fit with genetic algorithm <<<')
    
    def fit_with_pso(self, num_particle):
        print('    [3] >>> Fit with particle swarm optimization algorithm <<<')
        space = SpacePSO(num_particle, self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test,
                         self.y_test, Config.EPOCHS)
        space.particles = []
        for _ in range(space.num_particle):
            x, y, prediction, loss, graph = self.build_model()
            sess = tf.Session()

            trainable_variables = tf.trainable_variables()

            trainable_tensor = []
            for i, v in enumerate(trainable_variables):
                v_tensor = graph.get_tensor_by_name(v.name)
                _initiate_value = np.random.uniform(-1, 1, v_tensor.shape)
                sess.run(tf.assign(v_tensor, _initiate_value))
                trainable_tensor.append(v_tensor)
            saver = tf.train.Saver()
            space.particles.append(Particle(graph, sess))
        prediction = space.train()
        inversed_prediction = self.scaler.inverse_transform(prediction)
        inversed_prediction = np.asarray(inversed_prediction)
        self.y_test_inversed = self.scaler.inverse_transform(self.y_test)

        MAE_err = MAE(inversed_prediction, self.y_test_inversed)
        RMSE_err = np.sqrt(MSE(inversed_prediction, self.y_test_inversed))
        print("=== error = {}, {} ===".format(MAE_err, RMSE_err))

    def fit_with_woa(self, num_particle):
        print('    [3] >>> Fit with particle swarm optimization algorithm <<<')

    def fit_with_bp(self, optimizer, batch_size):
        self.x, self.y, self.outputs, self.new_state, self.prediction, self.loss, self.graph = self.build_model()
        print('    [3] >>> Fit with back propagation algorithm <<<')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=3e-4)
        self.optimize = self.optimizer.minimize(self.loss)
        cost_train_set = []
        cost_valid_set = []
        epoch_set = []

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess.run(init)
            print(">>> Start training with lstm <<<")
            for epoch in range(Config.EPOCHS):
                start_time = time.time()
                total_batch = int(len(self.x_train) / batch_size)
                avg_cost = 0
                for i in range(total_batch):
                    batch_xs = self.x_train[i * batch_size:(i + 1) * batch_size]
                    batch_ys = self.y_train[i * batch_size:(i + 1) * batch_size]
                    new_state = sess.run(self.new_state, feed_dict={self.x: batch_xs, self.y: batch_ys})
                    outputs = sess.run(self.outputs, feed_dict={self.x: batch_xs, self.y: batch_ys})
                    prediction = sess.run(self.prediction, feed_dict={self.x: batch_xs, self.y: batch_ys})

                    sess.run(self.optimize, feed_dict={self.x: batch_xs, self.y: batch_ys})
                    avg_cost += sess.run(self.loss, feed_dict={self.x: batch_xs, self.y: batch_ys}) / total_batch
                val_cost = sess.run(self.loss, feed_dict={self.x: self.x_valid, self.y: self.y_valid})
                cost_train_set.append(avg_cost)
                cost_valid_set.append(val_cost)

                # if self.early_stopping:
                #     if epoch > self.patience:
                #         if not self.early_stopping_desition(cost_train_set, self.patience):
                #             print(">>>Early stopping training<<<")
                #             break
                print('Epoch {}: cost = {} with time = {}'.format(epoch + 1, avg_cost, time.time() - start_time))
            print('>>> Training model done <<<')
            # draw training process loss
            # self.draw_train_loss(cost_train_set, cost_valid_set)
            # compute prediction and error to evaluate model
            prediction = sess.run(self.prediction, feed_dict={self.x: self.x_test})
            inversed_prediction = self.scaler.inverse_transform(prediction)
            inversed_prediction = np.asarray(inversed_prediction)
            self.y_test_inversed = self.scaler.inverse_transform(self.y_test)
            MAE_err, RMSE_err, r2_err = self.compute_error(inversed_prediction, self.y_test_inversed)
            print("=== error: MAE = {}, RMSE = {},R2 = {} ===".format(MAE_err, RMSE_err, r2_err))
            # save prediction
            # prediction_file = self.results_save_path + self.file_name + '.csv'
            # predictionDf = pd.DataFrame(np.array(inversed_prediction))
            # predictionDf.to_csv(prediction_file, index=False, header=None)
            # # save model
            # saver.save(sess, self.model_save_path + self.file_name + '/model')
            # summary = open(self.evaluation_path, 'a+')
            # summary.write('{}, {}, {}, {}\n'.format(self.file_name, MAE_err, RMSE_err, r2_err))
        print('    [3] >>> Fit with back propagation algorithm complete <<<')

    def fit_with_sfo(self, num_particle):
        print('    [3] >>> Fit with sf optimization algorithm <<<')

    def fit_with_isfo(self, num_particle):
        print('    [3] >>> Fit with isf optimization algorithm <<<')
