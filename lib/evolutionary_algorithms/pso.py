import time
import math

import numpy as np
import random
import tensorflow as tf
import multiprocessing as mp
from multiprocessing.pool import ThreadPool


class Particle:
    def __init__(self, graph=None, sess=None):

        self.graph = graph
        self.sess = sess

        self.x = self.graph.get_tensor_by_name('x:0')
        self.y = self.graph.get_tensor_by_name('y:0')
        self.prediction = self.graph.get_tensor_by_name('prediction:0')
        self.loss = self.graph.get_tensor_by_name('loss:0')

        self.trainable_variables = tf.trainable_variables()
        self.trainable_tensor = []
        # self.update_tensor_opts = []
        for i, v in enumerate(self.trainable_variables):
            v_tensor = self.graph.get_tensor_by_name(v.name)
            self.trainable_tensor.append(v_tensor)

        self.position = []
        self.velocity = []
        for v in tf.trainable_variables():
            v_value = self.sess.run(self.graph.get_tensor_by_name(v.name))
            self.position.append(v_value)
            self.velocity.append(np.zeros(v_value.shape))

        self.pbest_position = self.position
        self.velocity = np.array(self.velocity)
        self.pbest_position = np.array(self.pbest_position)
        self.position = np.array(self.position)
        self.pbest_value = float('inf')

    def __str__(self):
        print("My position: {} and pbest is: {}".format(self.graph, self.pbest_position))

    def fix_parameter_after_update(self):
        for i in range(len(self.position)):
            num_dimensional = len(self.position[i].shape)
            if num_dimensional == 1:
                for j in range(self.position[i].shape[0]):
                    if self.position[i][j] > 1:
                        self.position[i][j] = 1
                    elif self.position[i][j] < -1:
                        self.position[i][j] = -1
            elif num_dimensional == 2:
                for j in range(self.position[i].shape[0]):
                    for k in range(self.position[i].shape[1]):
                        if self.position[i][j][k] > 1:
                            self.position[i][j][k] = 1
                        elif self.position[i][j][k] < -1:
                            self.position[i][j][k] = -1

    def move(self):
        # self.fix_parameter_after_update()
        assignment_ops = []
        for i, v_tensor in enumerate(self.trainable_tensor):
            assignment_ops.append(tf.assign(v_tensor, self.position[i]))
        self.sess.run(assignment_ops)
        tf.reset_default_graph()

    def predict(self, x):
        return self.sess.run(self.prediction, feed_dict={self.x: x})

    def fitness(self, x, y):
        fitness_value = self.sess.run(self.loss, feed_dict={self.x: x, self.y: y})
        fitness_value = np.around(fitness_value, decimals=7)
        return fitness_value


class SpacePSO:
    def __init__(self, num_particle=50, x_train=None, y_train=None, x_valid=None, y_valid=None, x_test=None,
                 y_test=None, epochs=None):

        self.num_particle = num_particle
        self.particles = []
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs
        self.particles = []
        self.gbest_value = float('inf')
        self.gbest_paticle = None
        self.max_w_old_velocation = 0.9
        self.min_w_old_velocation = 0.4
        self.w_local_best_position = 1.2
        self.w_global_best_position = 1.2

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()

    def set_gbest(self):

        for particle in self.particles:
            fitness_cadidate = particle.fitness(self.x_train, self.y_train)
            if particle.pbest_value > fitness_cadidate:
                particle.pbest_value = fitness_cadidate
                particle.pbest_position = particle.position

            if self.gbest_value > fitness_cadidate:
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                self.gbest_paticle = particle

    def move_particles(self):
        for particle in self.particles:
            r1 = np.random.random_sample()
            r2 = np.random.random_sample()

            change_base_on_old_velocity = self.w_old_velocation * particle.velocity
            change_base_on_local_best = self.w_local_best_position * r1 * (particle.pbest_position - particle.position)
            change_base_on_global_best = self.w_global_best_position * r2 * (self.gbest_position - particle.position)

            new_velocity = change_base_on_old_velocity + change_base_on_local_best + change_base_on_global_best

            particle.velocity = new_velocity
            # assign new value for trainables value to be nearer optimize global
            particle.position = particle.position + particle.velocity
            particle.move()

    def early_stopping(self, array, patience=20):
        if patience <= len(array) - 1:
            value = array[len(array) - patience]
            arr = array[len(array) - patience + 1:]
            check = 0
            for val in arr:
                if val < value:
                    check += 1
            if check != 0:
                return False
            return True
        raise ValueError

    def check_most_n_value(self, cost_train_set):
        check = 0
        for i in range(len(cost_train_set) - 2, len(cost_train_set) - 6, -1):
            if cost_train_set[i] == cost_train_set[-1]:
                check += 1
            if check == 4:
                return True
        return False

    def train(self):
        cost_train_set = []
        cost_valid_set = []
        epoch_set = []
        epoch = 200
        for iteration in range(epoch):
            self.w_old_velocation = (epoch - iteration) / epoch * (self.max_w_old_velocation - self.min_w_old_velocation) + self.min_w_old_velocation
            start_time = time.time()
            self.set_gbest()
            self.move_particles()
            cost_train_set.append(round(self.gbest_value, 5))
            cost_valid_set.append(self.gbest_paticle.fitness(self.x_valid, self.y_valid))
            training_history = 'iteration inference: %d fitness = %.8f with time for running: %.2f '\
                % (iteration, self.gbest_value, time.time() - start_time)
            print(training_history)
            if len(cost_train_set) > 20:
                if self.early_stopping(cost_train_set):
                    print('[X] -> Early stoping because the profit is not increase !!!')
                    break

        print("The best solution in iterations: {} has fitness = {} in train set".format(iteration, self.gbest_value))
        return self.gbest_paticle, self.gbest_paticle.predict(self.x_test), cost_train_set, cost_valid_set
