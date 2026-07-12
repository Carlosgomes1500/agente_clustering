# -*- coding: utf-8 -*-
from sklearn.cluster import AgglomerativeClustering
from algorithms.base import BaseClusteringWrapper

class HierarchicalWrapper(BaseClusteringWrapper):
    """Wrapper para padronizar a execução do Hierarchical (Ward)"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
