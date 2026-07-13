# -*- coding: utf-8 -*-
import numpy as np
from algorithms.base import BaseClusteringWrapper

class KMeansWrapper(BaseClusteringWrapper):
    """Implementação do K-Means do zero usando NumPy"""
    
    def __init__(self, n_clusters, random_state=42, max_iter=300, tol=1e-4):
        super().__init__(n_clusters, random_state)
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        
    def fit_predict(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        # 1. Inicialização de centroides aleatória
        random_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].copy()
        
        labels = np.zeros(n_samples, dtype=int)
        
        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()
            
            # 2. Atribuir pontos aos centroides mais próximos
            # Usando broadcasting: X[:, np.newaxis] (n, 1, d) - centroids (k, d) -> (n, k, d)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # 3. Recalcular os centroides (média dos pontos designados)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    self.centroids[k] = cluster_points.mean(axis=0)
            
            # 4. Verificar convergência
            shift = np.linalg.norm(self.centroids - old_centroids)
            if shift < self.tol:
                break
                
        return labels
