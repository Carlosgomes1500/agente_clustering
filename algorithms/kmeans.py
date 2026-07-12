# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
from algorithms.base import BaseClusteringWrapper

class KMeansWrapper(BaseClusteringWrapper):
    """Wrapper para padronizar a execução do KMeans"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
