import time
import random

import numpy as np
import tensorflow as tf


class WhaleSpace:
    def __init__(self, num_particle=100, x_train=None, y_train=None, x_valid=None, y_valid=None, x_test=None,
                 y_test=None, epochs=None):

        self.num_particle = num_particle
        self.particles = []

        self.x_train = np.concatenate((x_train, x_valid), axis=0)
        self.y_train = np.concatenate((y_train, y_valid), axis=0)
        # self.x_train = x_train
        # self.y_train = y_train

        # self.x_valid = x_valid
        # self.y_valid = y_valid

        self.x_test = x_test
        self.y_test = y_test

        self.epochs = epochs
        self.particles = []
        self.gbest_value = float('inf')

    def set_gbest(self):
        for particle in self.particles:
            fitness_cadidate = particle.fitness(self.x_train, self.y_train)
            # if particle.pbest_value > fitness_cadidate:
            #     particle.pbest_value = fitness_cadidate
            #     particle.pbest_position = particle.position

            if self.gbest_value > fitness_cadidate:
                self.gbest_value = fitness_cadidate
                self.gbest_position = particle.position
                self.gbest_paticle = particle

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

    def train(self):
        train_set = []
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.set_gbest()

            a = 2 * np.cos(epoch / (self.epochs - 1))

            for particle in self.particles:

                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                b = 1
                p = np.random.rand()
                # print('>>> Parameter value: a = {}, r = {}, A = {}, C = {}, l = {}, p = {}'.format(a, r, A, C, l, p))
                if p < 0.5:
                    # shrinking encircling mechanism
                    if np.abs(A) < 1:
                        # Update base on best agent
                        D = np.abs(C * self.gbest_position - particle.position)
                        particle.position = self.gbest_position - A * D
                    else:
                        # Update base on random agent
                        random_agent_idx = np.random.randint(0, len(self.particles))
                        random_particle = self.particles[random_agent_idx]
                        D = np.abs(C * random_particle.position - particle.position)
                        particle.position = random_particle.position - A * D
                else:
                    # Spiral updating position
                    D = np.abs(self.gbest_position - particle.position)
                    particle.position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.gbest_position

                particle.fix_parameter_after_update()
                particle.move()
            train_set.append(self.gbest_value)
            # infor = 'Number less: {}, number bigger: {}, number by best agent: {}, number by random agent: {}'\
            #     .format(_num_particle_by_less, len(self.particles) - _num_particle_by_less,
            #             _num_particle_by_less - _num_by_random_agent, _num_by_random_agent)
            training_history = 'Iteration {}, best fitness = {} with time = {}'\
                .format(epoch, train_set[-1], round(time.time() - epoch_start_time, 4))
            print(training_history)
            # if epoch > 20:
            #     if self.early_stopping(train_set):
            #         print('[X] -> Early stoping because the profit is not increase !!!')
                    # break
        return self.gbest_paticle.predict(self.x_test), self.gbest_paticle

    def random_number_differ_i(self, i, n):
        num = random.randint(1, n + 1)
        while num != i:
            return num

    def train_with_levy_flight_woa(self):
        train_set = []
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.set_gbest()

            a = 2 * np.cos(epoch / (self.epochs - 1))
            for i, particle in enumerate(self.particles):
                r = np.random.rand()
                A = 2 * a * r - a
                C = 2 * r
                l = np.random.uniform(-1, 1)
                muy = np.random.uniform(-1, 1)
                b = 1
                p = np.random.rand()
                sign = random.choice([-1, 0, 1])
                # print('>>> Parameter value: a = {}, r = {}, A = {}, C = {}, l = {}, p = {}'.format(a, r, A, C, l, p))

                if p < 0.5:
                    # shrinking encircling mechanism
                    if np.abs(A) < 1:
                        # Update base on best agent
                        D = np.abs(C * self.gbest_position - particle.position)
                        particle.position = self.gbest_position - A * D
                    else:
                        # Update base on random agent
                        random_agent_idx = np.random.randint(0, len(self.particles))
                        random_particle = self.particles[random_agent_idx]
                        D = np.abs(C * random_particle.position - particle.position)
                        particle.position = random_particle.position - A * D
                else:
                    # Spiral updating position
                    D = np.abs(self.gbest_position - particle.position)
                    particle.position = D * np.exp(b * l) * np.cos(2 * np.pi * l) + self.gbest_position
                particle.position = particle.position + muy * sign
                particle.fix_parameter_after_update()
                particle.move()
            train_set.append(self.gbest_value)
            # infor = 'Number less: {}, number bigger: {}, number by best agent: {}, number by random agent: {}'\
            #     .format(_num_particle_by_less, len(self.particles) - _num_particle_by_less,
            #             _num_particle_by_less - _num_by_random_agent, _num_by_random_agent)
            training_history = 'Iteration {}, best fitness = {} with time = {}'\
                .format(epoch, train_set[-1], round(time.time() - epoch_start_time, 4))
            print(training_history)
            # if epoch > 20:
            #     if self.early_stopping(train_set):
            #         print('[X] -> Early stoping because the profit is not increase !!!')
                    # break
        return self.gbest_paticle.predict(self.x_test), self.gbest_paticle