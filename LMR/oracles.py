import numpy as np
from scipy.misc import logsumexp
from scipy.special import expit
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

        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, l2_coef=0.001):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.lambda_2 = l2_coef
     
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
    
    
class MulticlassLogistic(BaseSmoothOracle):
    """
    Оракул для задачи многоклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    
    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """
    
    def __init__(self, class_number=None, l2_coef=0.001):
        """
        Задание параметров оракула.
        
        class_number - количество классов в задаче
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.lambda_2 = l2_coef
        self.class_number = class_number
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        if self.class_number == None:
            self.class_number = max(y) + 1
        A = X.dot(w.T)
        return (self.lambda_2 / 2) * (np.linalg.norm(w) ** 2) + (1 / X.shape[0]) * (logsumexp(A, axis=1).sum()) +\
               (-1 / X.shape[0]) * (A[y.reshape(len(y), 1) == np.arange(self.class_number)]).sum()
        
    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        if self.class_number == None:
            self.class_number = max(y) + 1
        s1 = X.T.dot((y.reshape(len(y), 1) == np.arange(self.class_number)))
        s2 = X.T.dot((1 / np.exp(X.dot(w.T) - np.amax(X.dot(w.T), axis=1).reshape(X.shape[0], 
            1)).sum(axis=1)).reshape(X.shape[0], 1)
            * np.exp(X.dot(w.T) - np.amax(X.dot(w.T), axis=1).reshape(X.shape[0], 1)))
        return (-1 / X.shape[0]) * s1.T + (1 / X.shape[0]) * s2.T + self.lambda_2 * w
