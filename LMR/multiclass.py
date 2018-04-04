class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        self.kwargs = kwargs
        self.classifier = classifier
        self.mode = mode
        
    def fit(self, X, y):
        """
        Обучение классификатора
        """
        self.classifier_set = []
        if self.mode == 'one_vs_all':
            for i in range(0, np.max(y) + 1):
                mask = y != i
                temp = y.copy()
                temp[mask] = -1
                temp_classifier = self.classifier(**self.kwargs)
                temp_classifier.fit(X, temp)
                self.classifier_set.append(temp_classifier)
        else:
            self.max = np.max(y)
            for i in range(self.max):
                l = []
                for j in range(i + 1, self.max + 1):
                    mask1 = y == i
                    mask2 = y == j
                    temp = y.copy()
                    temp[mask1] = 1
                    temp[mask2] = -1
                    l.append(self.classifier(**self.kwargs))
                    l[-1].fit(X[mask1 | mask2], temp[mask1 | mask2])
                self.classifier_set.append(l)
    
    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'one_vs_all':
            y = [cl.predict_proba(X)[:, 1] for cl in self.classifier_set]
            return np.array(y).argmax(axis=0)
        else:
            max_count = np.zeros((X.shape[0], 1)).astype(int)
            arr = np.empty((X.shape[0], 0))
            for i in range(self.max):
                for j in range(i, self.max):
                    pred = np.where(self.classifier_set[i][j - i].predict_proba(X)[:, 1] > 0.5, i, j + 1)
                    arr = np.hstack((arr, pred[:, np.newaxis]))
            res = np.zeros((X.shape[0], 1)).astype(int)
            max_count = np.zeros((X.shape[0], 1)).astype(int)
            for i in range(self.max + 1):
                temp = (arr == i).sum(axis=1).reshape((X.shape[0], 1))
                res[max_count < temp] = i
                max_count = np.maximum(max_count, temp)
            return res
