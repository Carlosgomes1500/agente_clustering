#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Orquestrador do Benchmark Unificado de Algoritmos de Clusterização
"""

import sys
import os
import argparse
import numpy as np
import warnings

# Ajusta path para rodar direto de /benchmarks/ ou da raiz
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings('ignore')

from benchmarks.datasets import get_datasets
from benchmarks.algorithms import run_all_algorithms
from benchmarks.results import compile_results
from utils.data_processing import preprocess_data

def process_single_dataset(X, y, dataset_name, fast_mode=False):
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
            return []
        
        dataset_results = run_all_algorithms(X_scaled, y_encoded, k, dataset_name, fast_mode=fast_mode)
        print(f"✓ {dataset_name} processado com sucesso!")
        return dataset_results
        
    except Exception as e:
        print(f"✗ Erro fatal ao processar {dataset_name}: {e}")
        return []

def main(run_mode='all'):
    print("\n" + "="*70)
    print(f"BENCHMARK UNIFICADO DE CLUSTERIZAÇÃO (Modo: {run_mode.upper()})")
    print("="*70)

    all_results = []

    for X, y, dataset_name in get_datasets(run_mode):
        fast_mode_for_algo = (run_mode == 'fast')
        res = process_single_dataset(X, y, dataset_name, fast_mode=fast_mode_for_algo)
        all_results.extend(res)

    compile_results(all_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Unificado de Clusterização")
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'fast', 'native', 'openml'],
                        help="Modo de execução (all=tudo, fast=reduzido/rápido, native=apenas iris/wine/digits, openml=apenas openml)")
    args = parser.parse_args()
    
    main(args.mode)
