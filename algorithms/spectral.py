# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import eigh
from algorithms.base import BaseClusteringWrapper
from algorithms.kmeans import KMeansWrapper

class SpectralWrapper(BaseClusteringWrapper):
    """Implementação do Spectral Clustering do zero usando NumPy/SciPy"""
    
    def __init__(self, n_clusters, random_state=42, gamma=1.0):
        super().__init__(n_clusters, random_state)
        self.gamma = gamma
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # 1. Matriz de Similaridade W usando RBF (Gaussian) Kernel
        # dist_sq tem shape (n, n) com todas as distâncias pareadas
        dist_sq = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
        W = np.exp(-self.gamma * dist_sq)
        np.fill_diagonal(W, 0) # Sem auto-loops para o algoritmo
        
        # 2. Matriz Laplaciana Normalizada Simétrica
        # D^(-1/2)
        d = np.sum(W, axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.fmax(d, 1e-12))
        
        # Como D é diagonal, multiplicar D^(-1/2) * W * D^(-1/2) pode ser otimizado
        # com broadcasting nas linhas e colunas
        W_norm = W * d_inv_sqrt[:, np.newaxis] * d_inv_sqrt[np.newaxis, :]
        
        # L_sym = I - W_norm
        I = np.eye(n_samples)
        L_sym = I - W_norm
        
        # 3. Calcular os Autovalores e Autovetores
        # eigh é otimizado para matrizes simétricas reais e retorna em ordem crescente
        eigenvalues, eigenvectors = eigh(L_sym)
        
        # Pegar os k primeiros autovetores correspondentes aos menores autovalores
        U = eigenvectors[:, :self.n_clusters]
        
        # Normalizar as linhas de U (algoritmo de Ng, Jordan, Weiss)
        norms = np.linalg.norm(U, axis=1, keepdims=True)
        U_normalized = U / np.fmax(norms, 1e-12)
        
        # 4. Agrupar as linhas usando o nosso K-Means recém-escrito
        kmeans = KMeansWrapper(n_clusters=self.n_clusters, random_state=self.random_state)
        return kmeans.fit_predict(U_normalized)
