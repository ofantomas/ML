import numpy as np
import scipy
import time


class PEGASOSMethod:
    """
    Реализация метода Pegasos для решения задачи svm.
    """

    def __init__(self, step_lambda, batch_size, num_iter, random_seed=None):
        """
        step_lambda - величина шага, соответствует 

        batch_size - размер батча

        num_iter - число итераций метода, предлагается делать константное
        число итераций 
        """
        self.batch_size = batch_size
        self.step_lambda = step_lambda
        self.num_iter = num_iter
        self.seed = random_seed
        self.w = None
        self.history = {}
        
    def get_params(self, deep=False):
        return {'step_lambda': self.step_lambda, 'num_iter': self.num_iter,
                'random_seed': self.seed, 'batch_size': self.batch_size}
    
    def set_params(self, **params):
        self.__dict__.update(params)
        return self

    def fit(self, X, y, trace=True, accuracy=False, log_freq=0.01):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        y = y.reshape(X.shape[0])
        
        if self.seed is None:
            self.seed = np.random.randint(0, 1000, size=1)[0]
        np.random.seed(self.seed)

        self.w = np.zeros((X.shape[1]))

        if accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, random_state=11)
            self.history['accuracy'] = [
                (self.predict(X_test) == y_test).sum() / y_test.shape[0]]
        else:
            X_train, y_train = X, y

        self.history['func'] = [self.get_objective(X_train, y_train)]
        self.history['epoch'] = [0]

        if trace is True:
            self.history['time'] = [0]

        f_min = self.get_objective(X_train, y_train)
        self.w_min = self.w.copy()

        curr_ep = 0
        coefs = np.random.choice(
            range(X_train.shape[0]), self.num_iter * self.batch_size)

        for i in range(self.num_iter):
            start = time.time()
            self.w = (1 - (1 / (i + 1))) * self.w + (1 / self.batch_size * (i + 1) * self.step_lambda) *\
                self.get_gradient(X_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]],
                                  y_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]])
            curr_ep = self.batch_size * (i + 1) / X.shape[0]
            if 1 / ((self.step_lambda ** (0.5)) * np.linalg.norm(self.w)) < 1:
                self.w *= 1 / ((self.step_lambda ** (0.5))
                               * np.linalg.norm(self.w))
            if (curr_ep - self.history['epoch'][-1] >= log_freq):
                cur_f = self.get_objective(X_train, y_train)
                if cur_f < f_min:
                    f_min = cur_f
                    self.w_min = self.w.copy()
                self.history['func'].append(cur_f)
                self.history['epoch'].append(curr_ep)
                if accuracy is True:
                    self.history['accuracy'].append(
                        (self.predict(X_test) == y_test).sum() / y_test.shape[0])
                if trace is True:
                    self.history['time'].append(
                        time.time() - start + self.history['time'][-1])
        if trace is True:
            self.history['sol'] = (f_min, self.w_min)
        return self.history if (trace is True) else (f_min, self.w_min)

    def get_objective(self, X, y):
        M = 1 - y * X.dot(self.w.T)
        return (self.step_lambda / 2) * (self.w ** 2).sum() + (1 / X.shape[0]) * np.where(M > 0, M, 0).sum()

    def get_gradient(self, X, y):
        M = y * X.dot(self.w.T)
        return (np.where(M < 1, 1, 0)[:, np.newaxis] * y[:, np.newaxis] * X).sum(axis=0)

    def predict(self, X):
        """
        Получить предсказания по выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        """
        if self.w is None:
            raise TypeError('Weights have not been computed yet')
        return np.sign(X.dot(self.w_min))


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss='hinge', step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, C=1):
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss = loss
        self.C = C
        self.history = {}
        self.w = None

    def fit(self, X, y, w_0=None, trace=True, accuracy=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        y = y.reshape(X.shape[0])
        
        if self.loss == 'logistic':
            self.cl = oracles.BinaryLogistic(self.C)
        else:
            self.cl = oracles.BinaryHinge(self.C)

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        if accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, random_state=11)
            self.history['accuracy'] = [
                (self.predict(X_test) == y_test).sum() / y_test.shape[0]]
        else:
            X_train, y_train = X, y

        if trace is True:
            self.history['time'] = [0]
            self.history['func'] = [self.get_objective(X_train, y_train)]

        f_min = self.get_objective(X_train, y_train)
        self.w_min = self.w.copy()

        for i in range(self.max_iter):
            start = time.time()
            prev_f = self.get_objective(X_train, y_train)
            self.w = self.w - (self.step_alpha / ((i + 1) **
                                                  (self.step_beta))) * self.get_gradient(X_train, y_train)
            cur_f = self.get_objective(X_train, y_train)
            if cur_f < f_min:
                f_min = cur_f
                self.w_min = self.w.copy()
            if accuracy is True:
                self.history['accuracy'].append(
                    (self.predict(X_test) == y_test).sum() / y_test.shape[0])
            if trace is True:
                self.history['func'].append(cur_f)
                self.history['time'].append(
                    time.time() - start + self.history['time'][-1])
            if abs(prev_f - cur_f) < self.tolerance:
                print('converged')
                break
        if trace is True:
            self.history['sol'] = (f_min, self.w_min)
        return self.history if (trace is True) else (f_min, self.w_min)

    def get_params(self, deep=False):
        return {'step_alpha': self.step_alpha, 'step_beta': self.step_beta,
                'max_iter' self.max_iter, 'tolerance': self.tolerance, 'C': self.C}
    
    def set_params(self, **params):
        self.__dict__.update(params)
        return self
    
    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        if self.w is None:
            raise TypeError('Weights have not been computed yet')
        return np.sign(X.dot(self.w_min)).reshape(X.shape[0])

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.cl.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.cl.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w_min


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(self, loss='hinge', batch_size=1, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=1000, random_seed=153, C=1):
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.C = C
        self.loss = loss
        self.history = {}
        self.seed = random_seed
        self.w = None

    def get_params(self, deep=False):
        return {'step_alpha': self.step_alpha, 'step_beta': self.step_beta, 
                'max_iter': self.max_iter, 'tolerance': self.tolerance, 'C': self.C,
                'random_seed': self.seed, 'batch_size': self.batch_size}
    
    def fit(self, X, y, w_0=None, trace=True, log_freq=1, accuracy=False):
        y = y.reshape(X.shape[0])
        
        np.random.seed(self.seed)

        if self.loss == 'logistic':
            self.cl = oracles.BinaryLogistic(self.C)
        else:
            self.cl = oracles.BinaryHinge(self.C)

        if w_0 is None:
            self.w = np.zeros(X.shape[1])
        else:
            self.w = w_0

        if accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=0.7, random_state=11)
            self.history['accuracy'] = [
                (self.predict(X_test) == y_test).sum() / y_test.shape[0]]
        else:
            X_train, y_train = X, y

        self.history['func'] = [self.get_objective(X_train, y_train)]
        self.history['epoch'] = [0]

        if trace is True:
            self.history['time'] = [0]
            self.history['weights_diff'] = [0]

        curr_ep = 0
        coefs = np.random.choice(
            range(X_train.shape[0]), self.max_iter * self.batch_size)
        f_min = self.get_objective(X_train, y_train)
        self.w_min = self.w.copy()

        for i in range(self.max_iter):
            start = time.time()
            self.w = self.w - (self.step_alpha / ((i + 1) ** (self.step_beta))) *\
                self.get_gradient(X_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]],
                                  y_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]])
            curr_ep = self.batch_size * (i + 1) / X.shape[0]
            if (curr_ep - self.history['epoch'][-1] >= log_freq):
                cur_f = self.get_objective(X_train, y_train)
                if cur_f < f_min:
                    f_min = cur_f
                    self.w_min = self.w.copy()
                self.history['func'].append(cur_f)
                self.history['epoch'].append(curr_ep)
                if accuracy is True:
                    self.history['accuracy'].append(
                        (self.predict(X_test) == y_test).sum() / y_test.shape[0])
                if trace is True:
                    self.history['time'].append(
                        time.time() - start + self.history['time'][-1])
                if abs(self.history['func'][-2] - self.history['func'][-1]) < self.tolerance:
                    print('converged')
                    break
        if trace is True:
            self.history['sol'] = (f_min, self.w_min)
        return self.history if (trace is True) else (f_min, self.w_min)
