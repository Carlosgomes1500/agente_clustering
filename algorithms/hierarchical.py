# -*- coding: utf-8 -*-
from sklearn.cluster import AgglomerativeClustering

class HierarchicalWrapper:
    """Wrapper para padronizar a execução do Hierarchical (Ward)"""
    
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
