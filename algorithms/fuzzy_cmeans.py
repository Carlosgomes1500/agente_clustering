# -*- coding: utf-8 -*-
import numpy as np
from algorithms.base import BaseClusteringWrapper

class FuzzyCMeansWrapper(BaseClusteringWrapper):
    """Implementação do Fuzzy C-Means do zero usando NumPy"""
    
    def __init__(self, n_clusters, random_state=42, m=2.0, error=1e-4, maxiter=300):
        super().__init__(n_clusters, random_state)
        self.m = m
        self.error = error
        self.maxiter = maxiter
        
    def fit_predict(self, X):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        c = self.n_clusters
        m = self.m
        
        # 1. Inicializar matriz de pertinência U aleatoriamente
        # Dirichlet garante que a soma das probabilidades para cada ponto seja 1
        U = rng.dirichlet(np.ones(c), size=n_samples) 
        
        for iteration in range(self.maxiter):
            U_old = U.copy()
            
            # 2. Calcular os centroides (V) com base na pertinência (U^m)
            Um = U ** m
            # Um.T (c, n) @ X (n, d) -> centroides (c, d)
            centroids = (Um.T @ X) / Um.sum(axis=0)[:, np.newaxis]
            
            # 3. Calcular distâncias ao quadrado entre pontos e centroides
            # X[:, np.newaxis] (n, 1, d) e centroids (c, d) -> dist_sq (n, c)
            dist_sq = np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2)
            
            # Evitar divisão por zero
            dist_sq = np.fmax(dist_sq, np.finfo(np.float64).eps)
            
            # 4. Recalcular a matriz U
            # Equação: U_ij = 1 / sum_k ( d_ij / d_ik ) ^ (2/(m-1))
            power = 1.0 / (m - 1)
            # dist_sq[:, :, np.newaxis] (n, c, 1) / dist_sq[:, np.newaxis, :] (n, 1, c) -> (n, c, c)
            temp = (dist_sq[:, :, np.newaxis] / dist_sq[:, np.newaxis, :]) ** power
            U = 1.0 / np.sum(temp, axis=2)
            
            # 5. Verificar convergência
            if np.linalg.norm(U - U_old) < self.error:
                break
                
        # O cluster previsto é o que possui maior grau de pertinência (Hard Clustering no final)
        return np.argmax(U, axis=1)
