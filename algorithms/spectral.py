# -*- coding: utf-8 -*-
from sklearn.cluster import SpectralClustering
from algorithms.base import BaseClusteringWrapper

class SpectralWrapper(BaseClusteringWrapper):
    """Wrapper para padronizar a execução do Spectral Clustering"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        self.model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='nearest_neighbors')
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
