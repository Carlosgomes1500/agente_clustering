#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LARFSOM - Linear Adaptive Resonance Theory Fuzzy Self-Organizing Map
Algoritmo customizado de clusterização baseado em redes neurais
"""

import numpy as np
from sklearn.metrics import pairwise_distances


class LARFSOM:
    """
    Linear Adaptive Resonance Theory Fuzzy Self-Organizing Map
    
    Implementação de rede neural auto-organizável com atualização apenas da 
    Best Matching Unit (BMU) usando decaimento linear da taxa de aprendizado.
    
    Parâmetros:
    -----------
    n_clusters : int
        Número de neurônios (clusters)
    epochs : int, default=50
        Número de épocas de treinamento
    initial_lr : float, default=0.1
        Taxa de aprendizado inicial
    random_state : int, default=42
        Seed para reprodutibilidade
    """
    
    def __init__(self, n_clusters, epochs=50, initial_lr=0.1, random_state=42):
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.random_state = random_state
        self.weights = None
        self.labels_ = None
        
        np.random.seed(random_state)
    
    def fit(self, X):
        """
        Treina o LARFSOM nos dados X
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de treinamento
        
        Retorna:
        --------
        self : object
            Retorna a instância do modelo
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos aleatoriamente a partir das amostras
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.weights = X[random_indices].copy()
        
        # Treinamento
        for epoch in range(self.epochs):
            # Decaimento linear da taxa de aprendizado
            current_lr = self.initial_lr * (1 - epoch / self.epochs)
            
            # Embaralhar dados
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                sample = X[idx]
                
                # Encontrar Best Matching Unit (BMU)
                distances = np.linalg.norm(self.weights - sample, axis=1)
                bmu_idx = np.argmin(distances)
                
                # Atualizar apenas o BMU (Linear ART)
                self.weights[bmu_idx] += current_lr * (sample - self.weights[bmu_idx])
        
        return self
    
    def predict(self, X):
        """
        Prediz o rótulo de cluster para cada amostra
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados para predizer
        
        Retorna:
        --------
        labels : array, shape (n_samples,)
            Rótulos de cluster (0 a n_clusters-1)
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            sample = X[i]
            distances = np.linalg.norm(self.weights - sample, axis=1)
            labels[i] = np.argmin(distances)
        
        return labels
    
    def fit_predict(self, X):
        """
        Treina o modelo e retorna os rótulos de cluster
        
        Parâmetros:
        -----------
        X : array-like, shape (n_samples, n_features)
            Dados de treinamento
        
        Retorna:
        --------
        labels : array, shape (n_samples,)
            Rótulos de cluster (0 a n_clusters-1)
        """
        self.fit(X)
        self.labels_ = self.predict(X)
        return self.labels_
