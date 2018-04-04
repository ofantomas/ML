import scipy
import numpy as np
import cvxopt


class SVMSolver:
    """
    Класс с реализацией SVM через метод внутренней точки.
    """

    def __init__(self, C=1, method='primal', kernel='linear', degree=1, gamma=1):
        """
        C - float, коэффициент регуляризации

        method - строка, задающая решаемую задачу, может принимать значения:
            'primal' - соответствует прямой задаче
            'dual' - соответствует двойственной задаче
        kernel - строка, задающая ядро при решении двойственной задачи
            'linear' - линейное
            'polynomial' - полиномиальное
            'rbf' - rbf-ядро
        Обратите внимание, что часть функций класса используется при одном методе решения,
        а часть при другом
        """
        self.C = C
        self.method = method
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.dual = None
        self.ksi = None
        self.w0 = None
        self.w = None

    def get_params(self, deep=False):
        return {'C': self.C, 'method': self.method, 'kernel': self.kernel, 'degree': self.degree, 'gamma': self.gamma}
    
    def set_params(self, **kwargs):
        self.__dict__.update(kwargs)
        return self
        
    def K(self, X, Y):
        if self.kernel == 'linear':
            return X.dot(Y.T)
        elif self.kernel == 'polynomial':
            return (1 + X.dot(Y.T)) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * ((X ** 2).sum(axis=1)[:, np.newaxis] + (Y.T ** 2).sum(axis=0)[np.newaxis, :]
                                         - 2 * X.dot(Y.T)))
        else:
            raise TypeError('Unknown kernel type')

    def compute_primal_objective(self, X, y):
        """
        Метод для подсчета целевой функции SVM для прямой задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        if self.method == 'dual':
            raise TypeError('Cannot compute primal oblective for dual method')
        M = 1 - y.reshape(X.shape[0], 1) * X.dot(self.w)
        ksi = np.where(M > 0, M, 0).sum()
        return ksi * (self.C / X.shape[0]) + (1 / 2) * (self.w ** 2).sum()

    def compute_dual_objective(self, X, y):
        """
        Метод для подсчёта целевой функции SVM для двойственной задачи

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        """
        if self.method == 'primal':
            raise TypeError('Cannot compute dual oblective for primal method')
        if self.dual is None:
            raise TypeError("Function has not been optimized yet")
        return -self.dual.sum() + (1 / 2) * (np.outer(self.sv_y, self.sv_y)
                                             * self.K(self.sv, self.sv)).dot(self.dual).dot(self.dual)

    def fit(self, X, y, tolerance=1e-8, max_iter=100):
        """
        Метод для обучения svm согласно выбранной в method задаче

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        y - переменная типа numpy.array, правильные ответы на обучающей выборке,
        tolerance - требуемая точность для метода обучения
        max_iter - максимальное число итераций в методе

        """
        y = y.reshape(X.shape[0])
        opts = {'reltol': tolerance, 'maxiters': max_iter, 'show_progress': False}
        if self.method == 'primal':
            Y = np.tile(y, (X.shape[1] + 1, 1)).T
            G = np.hstack((np.ones((X.shape[0], 1)), X)) * (-Y)
            G = np.hstack((G, -np.identity(X.shape[0])))
            G = cvxopt.matrix(np.vstack((G, np.hstack(
                (np.zeros((X.shape[0], X.shape[1] + 1)), -np.identity(X.shape[0]))))))
            P = np.vstack(
                (np.eye(X.shape[1]), np.zeros((X.shape[0], X.shape[1]))))
            P = np.hstack((P, np.zeros((X.shape[0] + X.shape[1], X.shape[0]))))
            P = np.vstack((np.zeros((1, X.shape[1] + X.shape[0])), P))
            P = cvxopt.matrix(
                np.hstack((np.zeros((X.shape[0] + X.shape[1] + 1, 1)), P)))
            q = cvxopt.matrix(np.vstack(
                (np.zeros((X.shape[1] + 1, 1)), (self.C / X.shape[0]) * np.ones((X.shape[0], 1)))))
            h = cvxopt.matrix(
                np.vstack((-np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1)))))
            solution = cvxopt.solvers.qp(P, q, G, h, options=opts)
            self.w0 = solution['x'][0]
            self.w = np.array(solution['x'][1:X.shape[1] + 1])
        elif self.method == 'dual':
            P = cvxopt.matrix(self.K(X, X) * np.outer(y, y))
            q = cvxopt.matrix(-np.ones((X.shape[0], 1)))
            G = cvxopt.matrix(
                np.vstack((np.identity(X.shape[0]), -np.identity(X.shape[0]))))
            h = cvxopt.matrix(np.vstack(
                ((self.C / X.shape[0]) * np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1)))))
            A = cvxopt.matrix(y.reshape((1, y.shape[0])).astype(float))
            b = cvxopt.matrix([0.0])
            solution = cvxopt.solvers.qp(P, q, G, h, A, b, options=opts)
            lambda_1 = np.ravel(solution['x'])
            self.dual = lambda_1[lambda_1 > 1e-8]
            self.sv = X[lambda_1 > 1e-8]
            self.sv_y = y[lambda_1 > 1e-8]
            if self.kernel == 'linear':
                self.w = ((self.sv_y * self.dual)
                          [:, np.newaxis] * self.sv).sum(axis=0)
            ind = self.dual + 1e-8 < self.C / X.shape[0]
            if (ind.sum() == 0):
                self.w0 = 0
            else:
                self.w0 = (1 / ind.sum()) * (-self.K(self.sv, self.sv)
                                             [ind].dot(self.sv_y * self.dual) + self.sv_y[ind]).sum()
        else:
            raise TypeError('Unknown method')
        return self

    def predict(self, X):
        """
        Метод для получения предсказаний на данных

        X - переменная типа numpy.array, признаковые описания объектов из обучающей выборки
        """
        if self.w is None:
            if self.method == 'dual' and self.kernel != 'linear':
                return np.sign((self.K(X, self.sv) * (self.dual * self.sv_y)).sum(axis=1) + self.w0)
            raise TypeError('Weights have not been computed yet')
        else:
            return np.sign(X.dot(self.w) + self.w0)

    def get_w(self, X=None, y=None):
        """
        Получить прямые переменные (без учёта w_0)

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 

        return: одномерный numpy array
        """
        if self.w is None:
            if self.method == 'dual' and self.kernel != 'linear':
                raise TypeError(
                    'Cannot return weights when method is dual and kernel is not linear')
            else:
                raise TypeError('Weights have not been computed yet')
        else:
            return self.w

    def get_w0(self, X=None, y=None):
        """
        Получить вектор сдвига

        Если method = 'dual', а ядро линейное, переменные должны быть получены
        с помощью выборки (X, y) 

        return: float
        """
        if self.w0 is None:
            raise TypeError('w0 has not been computed yet')
        else:
            return self.w0

    def get_dual(self):
        """
        Получить двойственные переменные

        return: одномерный numpy array
        """
        if self.method == 'primal':
            raise TypeError(
                'Dual variables are unavailiable when method is "primal"')
        if self.dual is None:
            raise TypeError('Dual variables have not been computed yet')
        else:
            return self.dual
