# -*- coding: utf-8 -*-
from sklearn.cluster import SpectralClustering

class SpectralWrapper:
    """Wrapper para padronizar a execução do Spectral Clustering"""
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = SpectralClustering(n_clusters=n_clusters, random_state=42, affinity='nearest_neighbors')
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
