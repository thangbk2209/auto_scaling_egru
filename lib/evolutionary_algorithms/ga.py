import time
import random
import numpy as np

from config import *
from lib.includes.utils import *


class GeneticAlgorithm:
    cross_over_rate = 0.9
    mutation_rate = 0.02

    def __init__(self, population_size, x_train=None, y_train=None, x_valid=None, y_valid=None, x_test=None,
                 y_test=None, epochs=None, best_parent_ratio=0.5):

        self.population_size = population_size
        self.best_parent_size = int(self.population_size * best_parent_ratio)
        self.x_train = np.concatenate((x_train, x_valid), axis=0)
        self.y_train = np.concatenate((y_train, y_valid), axis=0)
        self.x_test = x_test
        self.y_test = y_test
        self.epochs = epochs

    def pred_vector_to_label(self, y_pred):
        _, argmaxs = y_pred.max(dim=1)
        labels = [1 if arg == 0 else -1 for arg in argmaxs.numpy()]
        list_decision = find_index_same_value_subarray(labels)
        return list_decision

    def _calculate_pred_profit(self, y_pred, bid, ask):
        list_decision = self.pred_vector_to_label(y_pred)

        if list_decision[0][0] != 0:
            fee_for_change_decision = 2 * 0.092 * len(list_decision)
        else:
            fee_for_change_decision = 2 * 0.092 * (len(list_decision) - 1)

        pred_profit = 0
        for decision, start_tick, stop_tick in list_decision:
            if decision == 1:
                if stop_tick != len(bid):
                    pred_profit += bid[stop_tick] - ask[start_tick]
                else:
                    pred_profit += bid[stop_tick] - ask[start_tick]
            else:
                if stop_tick != len(bid) - 1:
                    pred_profit += bid[start_tick] - ask[stop_tick + 1]
                else:
                    pred_profit += bid[start_tick] - ask[stop_tick]
        return pred_profit - fee_for_change_decision

    def calculate_pred_profit(self, solution):
        profit = 0
        for i in range(self.x_feature.shape[0]):
            _x_feature = self.x_feature[i]
            profit += self._calculate_pred_profit(solution(_x_feature), self.bids[i], self.asks[i])
        return profit

    def create_solution(self):
        solution = DeepTickSecondPytorchModel().double()
        # initiate first parameter base on uniform random distribution
        for var in solution.parameters():
            _initiate_value = np.random.uniform(-1, 1, var.shape)
            _initiate_value = torch.from_numpy(_initiate_value)
            var.data = _initiate_value

        fitness = self.calculate_pred_profit(solution)
        return [solution, fitness]

    def wheel_select(self, pop, prob):
        r = np.random.random()
        sum = prob[0]
        for i in range(1, len(pop) + 1):
            if sum > r:
                return i - 1
            else:
                sum += prob[i]

    def cal_rank(self, pop):
        fit = []
        for i in range(len(pop)):
            fit.append(pop[i][1])
        arg_rank = np.array(fit).argsort()
        rank = [i / sum(range(1, len(pop) + 1)) for i in range(1, len(pop) + 1)]
        return rank

    def cross_over(self, dad_element, mom_element):
        dad_variables = dad_element[0].parameters()
        mom_variables = mom_element[0].parameters()
        r = np.random.random()
        if r < self.cross_over_rate:
            child1 = DeepTickSecondPytorchModel().double()
            child2 = DeepTickSecondPytorchModel().double()
            for dad_var, mom_var, child1_var, child2_var in zip(dad_variables, mom_variables, child1.parameters(),
                                                                child2.parameters()):
                if np.random.randint(0, 2) % 2 == 0:
                    child1_var.data = dad_var
                    child2_var.data = mom_var
                else:
                    child1_var.data = mom_var
                    child2_var.data = dad_var
            fit1 = self.calculate_pred_profit(child1)
            fit2 = self.calculate_pred_profit(child2)
            if fit1 < fit2:
                return [child1, fit1]
            else:
                return [child2, fit1]
        if dad_element[1] < mom_element[1]:
            return dad_element
        else:
            return mom_element

    def select(self, pop):
        new_pop = []
        sum_fit = 0
        for i in range(len(pop)):
            sum_fit += pop[0][1]
        while len(new_pop) < self.population_size:
            rank = self.cal_rank(pop)
            # prob = self.normalize_fitness(pop)
            dad_index = self.wheel_select(pop, rank)
            mom_index = self.wheel_select(pop, rank)
            while dad_index == mom_index:
                mom_index = self.wheel_select(pop, rank)
            dad = pop[dad_index]
            mom = pop[mom_index]
            new_sol1 = self.cross_over(dad, mom)
            new_pop.append(new_sol1)
            # new_pop.append(new_sol2)
        return new_pop

    def normalize_fitness(self, pop):
        beta = 1
        _, max_fit = max(pop, key=lambda x: x[1])
        exp_fit = [np.exp(x[1] / (1650 * np.abs(max_fit))) for x in pop]
        prob = [exp_fit[i] / sum(exp_fit) for i in range(len(exp_fit))]
        return prob

    def mutate(self, pop):
        for i in range(len(pop)):
            for var in pop[i][0].parameters():
                if np.random.random() < self.mutation_rate:
                    var.data = torch.rand_like(var)
            pop[i][1] = self.calculate_pred_profit(pop[i][0])
        return pop

    def evolve(self):
        print('>>> Start evolve <<<')

        print('>>> Start create solution <<<')
        pop = [self.create_solution() for _ in range(self.population_size)]
        print('>>> Create solution complete <<<')
        gbest = pop[0]
        for iter in range(self.epochs):
            start_time = time.time()
            print('>>> iteration {} ... '.format(iter))
            pop = self.select(pop)
            pop = self.mutate(pop)
            best_fit = max(pop, key=lambda x: x[1])
            if best_fit[1] > gbest[1]:
                gbest = best_fit
            print("best current fit {}, best fit so far {}, iter {}".format(best_fit[1], gbest[1], iter))
            print(' Time for running: {}'.format(time.time() - start_time))
        return gbest
