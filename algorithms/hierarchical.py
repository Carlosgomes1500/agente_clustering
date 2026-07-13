# -*- coding: utf-8 -*-
import numpy as np
from algorithms.base import BaseClusteringWrapper

class HierarchicalWrapper(BaseClusteringWrapper):
    """Implementação do Agglomerative Clustering (Critério de Ward) do zero usando NumPy"""
    
    def __init__(self, n_clusters, random_state=42):
        super().__init__(n_clusters, random_state)
        
    def fit_predict(self, X):
        n_samples = X.shape[0]
        
        # Se for um dataset massivo, avisar que vai demorar (Complexidade O(N^3))
        if n_samples > 2000:
            print(f"\n[AVISO HIERÁRQUICO] Dataset grande detectado ({n_samples} amostras). O processamento O(N³) demorará muito sem o backend C do scikit-learn!")
        
        # 1. Matriz de distâncias iniciais
        # Para Ward, a distância inicial entre duas folhas é d(u,v) = 1/2 * ||u - v||^2
        dist_sq = np.sum((X[:, np.newaxis] - X[np.newaxis, :]) ** 2, axis=2)
        C = dist_sq * 0.5
        np.fill_diagonal(C, np.inf)
        
        # Cada amostra é seu próprio cluster no início
        labels = np.arange(n_samples)
        
        # O tamanho (quantidade de amostras) de cada cluster ativo
        sizes = np.ones(n_samples)
        
        # Lista para rastrear quais clusters ainda não foram fundidos
        active_clusters = list(range(n_samples))
        
        # 2. Loop de aglomeração: unir pares até sobrarem n_clusters
        for step in range(n_samples - self.n_clusters):
            # Encontrar o par (u, v) com o menor custo de fusão
            # np.argmin retorna um índice linear, unravel_index transforma em tupla 2D
            u, v = np.unravel_index(np.argmin(C), C.shape)
            
            # Garantir que u é menor que v só por padronização
            if u > v:
                u, v = v, u
                
            new_size = sizes[u] + sizes[v]
            
            # 3. Atualizar as distâncias do novo cluster fundido 'u' para todos os outros 'k' ativos
            # Usando a fórmula de Lance-Williams para o método Ward:
            for k in active_clusters:
                if k != u and k != v:
                    nk = sizes[k]
                    nu = sizes[u]
                    nv = sizes[v]
                    
                    # D(u U v, k) = ((n_u+n_k)*D(u,k) + (n_v+n_k)*D(v,k) - n_k*D(u,v)) / (n_u+n_v+n_k)
                    new_d = ((nu + nk) * C[u, k] + (nv + nk) * C[v, k] - nk * C[u, v]) / (nu + nv + nk)
                    C[u, k] = new_d
                    C[k, u] = new_d
                    
            # 4. Inutilizar o cluster 'v' (ele foi devorado por 'u')
            sizes[u] = new_size
            active_clusters.remove(v)
            
            C[v, :] = np.inf
            C[:, v] = np.inf
            C[u, u] = np.inf # Sem auto-loops
            
            # 5. Atualizar os labels: todos os pontos do antigo cluster 'v' recebem label 'u'
            labels[labels == v] = u
            
        # 6. Re-mapear os labels para o formato contínuo 0 a k-1
        unique_labels = np.unique(labels)
        final_labels = np.zeros_like(labels)
        for idx, cl in enumerate(unique_labels):
            final_labels[labels == cl] = idx
            
        return final_labels
