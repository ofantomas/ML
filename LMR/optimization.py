import numpy as np
import time
import oracles


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.history = {}
        self.w = None
 
    def fit(self, X, y, w_0=None, trace=False, accuracy=False):
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
        if self.loss_function == 'binary_logistic':
            self.cl = oracles.BinaryLogistic(**self.kwargs)
        else:
            self.cl = oracles.MulticlassLogistic(**self.kwargs)  
        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                self.w = np.zeros(X.shape[1])
            else:
                self.w = np.zeros((self.cl.class_number, X.shape[1]))
        else:
            self.w = w_0
        if accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=11)
            self.history['accuracy'] = [(self.predict(X_test) == y_test).sum() / y_test.shape[0]]
        else:
            X_train, y_train = X, y
        if trace is True:
            self.history['time'] = [0]
            self.history['func'] = [self.get_objective(X_train, y_train)]
        for i in range(self.max_iter):
            start = time.time()
            prev_f = self.get_objective(X_train, y_train)
            self.w = self.w - (self.step_alpha/((i + 1) ** (self.step_beta))) * self.get_gradient(X_train, y_train)
            if accuracy is True:
                self.history['accuracy'].append((self.predict(X_test) == y_test).sum() / y_test.shape[0])
            if trace is True:
                self.history['func'].append(self.get_objective(X_train, y_train))
                self.history['time'].append(time.time() - start + self.history['time'][-1])
            if abs(prev_f - self.get_objective(X, y)) < self.tolerance:
                break
        return self.history if (trace is True) else None
    
    def predict(self, X, threshold=0.5):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        if self.loss_function == 'binary_logistic':
            return np.where(self.predict_proba(X)[:, 1] > threshold, 1, -1)
        else:
            return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        if self.loss_function == 'binary_logistic':
            proba = scipy.special.expit(X.dot(self.w))
            return np.vstack((1 - proba, proba)).T
        else:
            return (1 / np.exp(X.dot(self.w.T) - np.amax(X.dot(self.w.T), axis=1).reshape(X.shape[0], 1))
                    .sum(axis=1)).reshape(X.shape[0], 1) * np.exp(X.dot(self.w.T) -
                    np.amax(X.dot(self.w.T), axis=1).reshape(X.shape[0], 1))
            
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
        return self.w
        

class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size=1, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        self.history = {}
        self.random_seed = random_seed
        self.w = None
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1, accuracy=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if self.loss_function == 'binary_logistic':
            self.cl = oracles.BinaryLogistic(**self.kwargs)
        else:
            self.cl = oracles.MulticlassLogistic(**self.kwargs)  
        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                self.w = np.zeros(X.shape[1])
            else:
                self.w = np.zeros((self.cl.class_number, X.shape[1]))
        else:
            self.w = w_0
        if accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=11)
            self.history['accuracy'] = [(self.predict(X_test) == y_test).sum() / y_test.shape[0]]
        else:
            X_train, y_train = X, y
        self.history['func'] = [self.get_objective(X_train, y_train)]
        self.history['epoch'] = [0]
        if trace is True:
            self.history['time'] = [0]
            self.history['weights_diff'] = [0]
        curr_ep = 0
        coefs = np.random.choice(range(X_train.shape[0]), self.max_iter * self.batch_size)
        
        for i in range(self.max_iter):
            start = time.time()
            prev_w = self.w
            self.w = self.w - (self.step_alpha/((i + 1) ** (self.step_beta))) *\
                self.get_gradient(X_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]],
                                  y_train[coefs[i * self.batch_size:(i + 1) * self.batch_size:]])
            curr_ep = self.batch_size * (i + 1) / X.shape[0]
            if (curr_ep - self.history['epoch'][-1] >= log_freq):
                self.history['func'].append(self.get_objective(X_train, y_train))
                self.history['epoch'].append(curr_ep)
                if accuracy is True:
                    self.history['accuracy'].append((self.predict(X_test) == y_test).sum() / y_test.shape[0])
                if trace is True:
                    self.history['time'].append(time.time() - start + self.history['time'][-1])
                    self.history['weights_diff'].append(((prev_w - self.w) ** 2).sum())
                if abs(self.history['func'][-2] - self.history['func'][-1]) < self.tolerance:
                    break
        return self.history if (trace is True) else None
