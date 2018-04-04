import numpy as np
import scipy


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')

        
class BinaryHinge(BaseSmoothOracle):
    """
    Оракул для задачи двухклассового линейного SVM.
    """
    
    def __init__(self, C=1):
        """
        Задание параметров оракула.
        """
        self.C = C
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        M = 1 - y * X.dot(w.T)
        return (1 / 2) * (np.linalg.norm(w[1::]) ** 2) + (self.C / X.shape[0]) * np.where(M > 0, M, 0).sum()
        
    def grad(self, X, y, w):
        """
        Вычислить субградиент функционала в точке w на выборке X с ответами y.
        Субгрдиент в точке 0 необходимо зафиксировать равным 0.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        w_ = np.append(0, w[1::])
        M = 1 - y * X.dot(w.T)
        return w_ - (self.C / X.shape[0]) * (np.where(M > 0, 1, 0)[:, np.newaxis] * (y[:, np.newaxis] * X)).sum(axis=0)


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, C=0.001):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.lambda_2 = C
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        return (1 / X.shape[0]) * (np.logaddexp(0, (-1) * y * X.dot(w)).sum(axis=0))\
            + (self.lambda_2 / 2) * (w ** 2).sum()
        
    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        s1 = scipy.special.expit((-y) * X.dot(w)) * (-y)
        return (1 / X.shape[0]) * X.T.dot(s1) + self.lambda_2 * w
