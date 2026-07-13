# -*- coding: utf-8 -*-
import gc
from algorithms.kmeans import KMeansWrapper
from algorithms.hierarchical import HierarchicalWrapper
from algorithms.spectral import SpectralWrapper
from algorithms.fuzzy_cmeans import FuzzyCMeansWrapper
from algorithms.larfsom import LARFSOM
from utils.evaluation import evaluate_algorithm

def run_all_algorithms(X_scaled, y_encoded, k, dataset_name, fast_mode=False):
    """
    Executa todos os algoritmos configurados para o dataset e retorna a lista de resultados.
    """
    dataset_results = []
    
    print(f"  Executando algoritmos...")
    
    res_kmeans = evaluate_algorithm(KMeansWrapper, 'K-Means', X_scaled, y_encoded, k, dataset_name, needs_subsample=False)
    if res_kmeans: dataset_results.append(res_kmeans)
        
    res_ward = evaluate_algorithm(HierarchicalWrapper, 'Hierárquico (Ward)', X_scaled, y_encoded, k, dataset_name, needs_subsample=False)
    if res_ward: dataset_results.append(res_ward)
        
    if not fast_mode:
        res_spectral = evaluate_algorithm(SpectralWrapper, 'Spectral', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
        if res_spectral: dataset_results.append(res_spectral)
            
        res_fuzzy = evaluate_algorithm(FuzzyCMeansWrapper, 'Fuzzy C-Means', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
        if res_fuzzy: dataset_results.append(res_fuzzy)
            
    res_larfsom = evaluate_algorithm(LARFSOM, 'LARFSOM', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
    if res_larfsom: dataset_results.append(res_larfsom)
    
    gc.collect()
    return dataset_results
