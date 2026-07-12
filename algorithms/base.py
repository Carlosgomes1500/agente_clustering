# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

class BaseClusteringWrapper(ABC):
    """
    Classe base para todos os wrappers de clusterização.
    Garante que todos implementem a mesma interface.
    """
    
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
        
    @abstractmethod
    def fit_predict(self, X):
        """
        Treina o modelo e retorna os rótulos preditos para X.
        """
        pass
