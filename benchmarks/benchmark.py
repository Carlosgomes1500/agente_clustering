#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark Unificado de Algoritmos de Clusterização
"""

import sys
import os
import gc
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml

from algorithms.kmeans import KMeansWrapper
from algorithms.hierarchical import HierarchicalWrapper
from algorithms.spectral import SpectralWrapper
from algorithms.fuzzy_cmeans import FuzzyCMeansWrapper
from algorithms.larfsom import LARFSOM
from utils.data_processing import preprocess_data
from utils.evaluation import evaluate_algorithm

results = []

def process_dataset(X, y, dataset_name, fast_mode=False):
    """Processa um dataset completo com tratamento de erros"""
    print(f"\n{'='*70}")
    print(f"Processando: {dataset_name}")
    print(f"{'='*70}")
    
    try:
        X_scaled, y_encoded = preprocess_data(X, y, dataset_name)
        k = len(np.unique(y_encoded))
        
        print(f"  Shape final: {X_scaled.shape}")
        print(f"  Número de classes: {k}")
        
        if k < 2:
            print(f"  ✗ Dataset inválido: menos de 2 classes!")
            return
        
        print(f"  Executando algoritmos...")
        
        res_kmeans = evaluate_algorithm(KMeansWrapper, 'K-Means', X_scaled, y_encoded, k, dataset_name, needs_subsample=False)
        if res_kmeans: results.append(res_kmeans)
            
        res_ward = evaluate_algorithm(HierarchicalWrapper, 'Hierárquico (Ward)', X_scaled, y_encoded, k, dataset_name, needs_subsample=False)
        if res_ward: results.append(res_ward)
            
        if not fast_mode:
            res_spectral = evaluate_algorithm(SpectralWrapper, 'Spectral', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
            if res_spectral: results.append(res_spectral)
                
            res_fuzzy = evaluate_algorithm(FuzzyCMeansWrapper, 'Fuzzy C-Means', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
            if res_fuzzy: results.append(res_fuzzy)
                
        res_larfsom = evaluate_algorithm(LARFSOM, 'LARFSOM', X_scaled, y_encoded, k, dataset_name, needs_subsample=True)
        if res_larfsom: results.append(res_larfsom)
        
        gc.collect()
        print(f"✓ {dataset_name} processado com sucesso!")
    except Exception as e:
        print(f"✗ Erro fatal ao processar {dataset_name}: {e}")

def run_benchmark(run_mode='all'):
    print("\n" + "="*70)
    print(f"BENCHMARK UNIFICADO DE CLUSTERIZAÇÃO (Modo: {run_mode.upper()})")
    print("="*70)

    fast_mode = (run_mode == 'fast')

    # FASE 1: Datasets Nativos
    if run_mode in ['all', 'fast', 'native']:
        print("\n[FASE 1] Datasets nativos (sklearn.datasets)\n")

        print("[Load] Carregando Iris...")
        try:
            iris = load_iris()
            process_dataset(iris.data, iris.target, 'Iris', fast_mode)
        except Exception as e:
            print(f"✗ Erro ao carregar Iris: {e}")

        print("\n[Load] Carregando Wine...")
        try:
            wine = load_wine()
            process_dataset(wine.data, wine.target, 'Wine', fast_mode)
        except Exception as e:
            print(f"✗ Erro ao carregar Wine: {e}")

        print("\n[Load] Carregando Digits...")
        try:
            digits = load_digits()
            process_dataset(digits.data, digits.target, 'Digits', fast_mode)
        except Exception as e:
            print(f"✗ Erro ao carregar Digits: {e}")

    # FASE 2: Datasets OpenML
    if run_mode in ['all', 'fast', 'openml']:
        print("\n" + "="*70)
        print("[FASE 2] Datasets OpenML\n")

        openml_datasets = [
            ('vehicle', 1, 'Vehicle'),
            ('letter', 1, 'Letter'),
            ('ionosphere', 1, 'Ionosphere'),
            ('usps', 1, 'USPS'),
        ]
        
        # Reduzir no modo fast
        if fast_mode:
            openml_datasets = openml_datasets[:2]

        for dataset_name, version, display_name in openml_datasets:
            print(f"\n[Load] Carregando {display_name}...")
            try:
                data = fetch_openml(name=dataset_name, version=version, as_frame='auto', parser='auto')
                process_dataset(data.data.values, data.target.values, display_name, fast_mode)
            except Exception as e:
                print(f"✗ Aviso ao carregar {display_name}: {str(e)[:100]}")

        if not fast_mode:
            print(f"\n[Load] Carregando Isolet...")
            try:
                isolet = fetch_openml(name='isolet', version=1, as_frame='auto', parser='auto')
                process_dataset(isolet.data.values, isolet.target.values, 'Isolet', fast_mode)
            except Exception as e:
                print(f"✗ Aviso ao carregar Isolet: {str(e)[:100]}")

            print(f"\n[Load] Carregando Waveform-5000...")
            try:
                waveform = fetch_openml(name='waveform-5000', version=1, as_frame='auto', parser='auto')
                process_dataset(waveform.data.values, waveform.target.values, 'Waveform-5000', fast_mode)
            except Exception as e:
                print(f"✗ Aviso ao carregar Waveform-5000: {str(e)[:100]}")

    # RESULTADOS FINAIS
    print("\n" + "="*70)
    print("RESULTADOS FINAIS")
    print("="*70 + "\n")

    if results:
        df_results = pd.DataFrame(results)
        print(df_results.to_markdown(index=False))
        
        # Salvar em CSV
        output_file = 'resultados_benchmark.csv'
        df_results.to_csv(output_file, index=False)
        
        print(f"\n✓ Benchmark concluído com sucesso!")
        print(f"Resultados salvos em: {output_file}")
        print(f"Total de execuções: {len(results)}")
        print(f"Datasets avaliados: {df_results['Dataset'].nunique()}")
    else:
        print("Nenhum resultado foi gerado.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Unificado de Clusterização")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'fast', 'native', 'openml'],
                        help="Modo de execução (all=tudo, fast=reduzido/rápido, native=apenas iris/wine/digits, openml=apenas openml)")
    args = parser.parse_args()
    
    run_benchmark(args.mode)
