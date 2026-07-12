# -*- coding: utf-8 -*-
import numpy as np
import skfuzzy as fuzz
from algorithms.base import BaseClusteringWrapper

class FuzzyCMeansWrapper(BaseClusteringWrapper):
    """Wrapper para padronizar a execução do Fuzzy C-Means (skfuzzy)"""
    
    def __init__(self, n_clusters, random_state=42, m=2.0, error=0.005, maxiter=1000):
        super().__init__(n_clusters, random_state)
        self.m = m
        self.error = error
        self.maxiter = maxiter
        
    def fit_predict(self, X):
        # fuzz.cmeans espera X transposto: shape (n_features, n_samples)
        cntr, u, _, _, _, _, _ = fuzz.cmeans(
            X.T, self.n_clusters, m=self.m, error=self.error, maxiter=self.maxiter
        )
        # u contém o grau de pertinência de cada amostra a cada cluster (shape: c, n)
        # O cluster previsto é o que possui maior grau
        return np.argmax(u, axis=0)
