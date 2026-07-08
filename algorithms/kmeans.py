# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans

class KMeansWrapper:
    """Wrapper para padronizar a execução do KMeans"""
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
