#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark Definitivo de Algoritmos de Clusterização
Com LARFSOM real e datasets selecionados (SEM CIFAR)
"""

import sys
import os
import gc

# Tentar importar e instalar se necessário
def install_package(package):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

packages_to_check = {
    'sklearn': 'scikit-learn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'packaging': 'packaging',
    'networkx': 'networkx',
    'skfuzzy': 'scikit-fuzzy',
    'tabulate': 'tabulate',
}

print("Verificando dependências...")
for import_name, pip_name in packages_to_check.items():
    try:
        __import__(import_name)
        print(f"  ✓ {pip_name}")
    except ImportError:
        print(f"  ✗ {pip_name} - Instalando...")
        install_package(pip_name)
        print(f"    ✓ {pip_name} instalado")

# Agora importar tudo
import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import skfuzzy as fuzz

# IMPORTAR LARFSOM REAL
try:
    from larfsom import LARFSOM
    print("  ✓ LARFSOM")
except ImportError:
    print("  ✗ LARFSOM não encontrado! Use: python -c 'from larfsom import LARFSOM'")
    sys.exit(1)

print("\nTodas as dependências carregadas com sucesso!\n")

results = []

def preprocess_data(X, y, dataset_name="Dataset"):
    """
    Pré-processamento robusto:
    - Converte csr_matrix para dense
    - Faz flatten para imagens multidimensionais
    - Remove/imputa NaNs
    - Codifica categóricas
    - Aplica StandardScaler
    - Codifica y com LabelEncoder
    """
    # Converter sparse matrix para dense
    if hasattr(X, 'toarray'):
        X = X.toarray()
    elif hasattr(X, 'todense'):
        X = np.array(X.todense())
    
    # Converter y para array
    y = np.array(y)
    
    # Flatten se for imagem multidimensional (ex: CIFAR, Digits)
    if len(X.shape) > 2:
        print(f"    Flattening imagem: {X.shape} → ", end="")
        X = X.reshape(X.shape[0], -1)
        print(f"{X.shape}")
    
    # Converter para DataFrame para facilitar processamento
    X = pd.DataFrame(X)
    
    # Remover linhas com NaNs
    mask = X.isnull().any(axis=1)
    if mask.any():
        print(f"    Removendo {mask.sum()} linhas com NaN")
        X = X[~mask].reset_index(drop=True)
        y = y[~mask]
    
    # Converter colunas categóricas para numéricas
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            except Exception as e:
                print(f"    Aviso ao codificar coluna {col}: {e}")
    
    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LabelEncoder para y (target)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    return X_scaled, y_encoded

def subsample_if_needed(X, y, max_samples=3000):
    """
    Subsampling estratificado para datasets grandes.
    Spectral Clustering tem O(N^3), então precisamos limitar.
    """
    if X.shape[0] > max_samples:
        print(f"    ⚠ Dataset grande ({X.shape[0]} > {max_samples}). Subsampling estratificado...")
        X_sub, _, y_sub, _ = train_test_split(
            X, y, 
            train_size=max_samples, 
            stratify=y, 
            random_state=42
        )
        print(f"    Reduzido para {X_sub.shape[0]} amostras")
        return X_sub, y_sub
    return X, y

def run_kmeans(X, y, k, dataset_name):
    """K-Means"""
    try:
        start = time()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        y_pred = km.fit_predict(X)
        elapsed = time() - start
        
        sil = silhouette_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        ari = adjusted_rand_score(y, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'K-Means',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        })
        print(f"    ✓ K-Means: {round(elapsed, 2)}s")
    except Exception as e:
        print(f"    ✗ K-Means: {e}")

def run_hierarchical(X, y, k, dataset_name):
    """Hierárquico Aglomerativo (ward)"""
    try:
        start = time()
        hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
        y_pred = hc.fit_predict(X)
        elapsed = time() - start
        
        sil = silhouette_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        ari = adjusted_rand_score(y, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'Hierárquico (Ward)',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        })
        print(f"    ✓ Hierárquico: {round(elapsed, 2)}s")
    except Exception as e:
        print(f"    ✗ Hierárquico: {e}")

def run_spectral(X, y, k, dataset_name):
    """Spectral Clustering com subsampling se necessário"""
    try:
        X_spec, y_spec = subsample_if_needed(X, y)
        
        start = time()
        sc = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
        y_pred = sc.fit_predict(X_spec)
        elapsed = time() - start
        
        # Se fizemos subsampling, usar os y_spec para métricas
        sil = silhouette_score(X_spec, y_pred)
        db = davies_bouldin_score(X_spec, y_pred)
        ari = adjusted_rand_score(y_spec, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'Spectral',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        })
        print(f"    ✓ Spectral: {round(elapsed, 2)}s")
    except Exception as e:
        print(f"    ✗ Spectral: {e}")

def run_fuzzy_cmeans(X, y, k, dataset_name):
    """Fuzzy C-Means com subsampling se necessário"""
    try:
        X_fuzzy, y_fuzzy = subsample_if_needed(X, y)
        
        start = time()
        cntr, u, _, _, _, _, _ = fuzz.cmeans(X_fuzzy.T, k, m=2.0, error=0.005, maxiter=1000)
        y_pred = np.argmax(u, axis=0)
        elapsed = time() - start
        
        sil = silhouette_score(X_fuzzy, y_pred)
        db = davies_bouldin_score(X_fuzzy, y_pred)
        ari = adjusted_rand_score(y_fuzzy, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'Fuzzy C-Means',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        })
        print(f"    ✓ Fuzzy C-Means: {round(elapsed, 2)}s")
    except Exception as e:
        print(f"    ✗ Fuzzy C-Means: {e}")

def run_larfsom(X, y, k, dataset_name):
    """LARFSOM Real (sem fallback)"""
    try:
        X_larf, y_larf = subsample_if_needed(X, y)
        
        start = time()
        larfsom_model = LARFSOM(n_clusters=k, epochs=50, initial_lr=0.1)
        y_pred = larfsom_model.fit_predict(X_larf)
        elapsed = time() - start
        
        sil = silhouette_score(X_larf, y_pred)
        db = davies_bouldin_score(X_larf, y_pred)
        ari = adjusted_rand_score(y_larf, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'LARFSOM',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        })
        print(f"    ✓ LARFSOM: {round(elapsed, 2)}s")
    except Exception as e:
        print(f"    ✗ LARFSOM: {e}")

def process_dataset(X, y, dataset_name):
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
        
        run_kmeans(X_scaled, y_encoded, k, dataset_name)
        run_hierarchical(X_scaled, y_encoded, k, dataset_name)
        run_spectral(X_scaled, y_encoded, k, dataset_name)
        run_fuzzy_cmeans(X_scaled, y_encoded, k, dataset_name)
        run_larfsom(X_scaled, y_encoded, k, dataset_name)
        
        gc.collect()  # Liberar memória
        print(f"✓ {dataset_name} processado com sucesso!")
    except Exception as e:
        print(f"✗ Erro fatal ao processar {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


print("\n" + "="*70)
print("BENCHMARK DEFINITIVO - 5 ALGORITMOS")
print("="*70)

# FASE 1: Datasets Nativos
print("\n[FASE 1] Datasets nativos (sklearn.datasets)\n")

print("[Load] Carregando Iris...")
try:
    iris = load_iris()
    process_dataset(iris.data, iris.target, 'Iris')
except Exception as e:
    print(f"✗ Erro ao carregar Iris: {e}")

print("\n[Load] Carregando Wine...")
try:
    wine = load_wine()
    process_dataset(wine.data, wine.target, 'Wine')
except Exception as e:
    print(f"✗ Erro ao carregar Wine: {e}")

print("\n[Load] Carregando Digits...")
try:
    digits = load_digits()
    process_dataset(digits.data, digits.target, 'Digits')
except Exception as e:
    print(f"✗ Erro ao carregar Digits: {e}")

# FASE 2: Datasets OpenML
print("\n" + "="*70)
print("[FASE 2] Datasets OpenML\n")

openml_datasets = [
    ('vehicle', 1, 'Vehicle'),
    ('letter', 1, 'Letter'),
    ('ionosphere', 1, 'Ionosphere'),
    ('usps', 1, 'USPS'),
]

for dataset_name, version, display_name in openml_datasets:
    print(f"\n[Load] Carregando {display_name}...")
    try:
        data = fetch_openml(name=dataset_name, version=version, as_frame='auto', parser='auto')
        process_dataset(data.data.values, data.target.values, display_name)
    except Exception as e:
        print(f"✗ Aviso ao carregar {display_name}: {str(e)[:100]}")

print(f"\n[Load] Carregando Isolet...")
try:
    isolet = fetch_openml(name='isolet', version=1, as_frame='auto', parser='auto')
    process_dataset(isolet.data.values, isolet.target.values, 'Isolet')
except Exception as e:
    print(f"✗ Aviso ao carregar Isolet: {str(e)[:100]}")

print(f"\n[Load] Carregando Waveform-5000...")
try:
    waveform = fetch_openml(name='waveform-5000', version=1, as_frame='auto', parser='auto')
    process_dataset(waveform.data.values, waveform.target.values, 'Waveform-5000')
except Exception as e:
    print(f"✗ Aviso ao carregar Waveform-5000: {str(e)[:100]}")

# RESULTADOS FINAIS
print("\n" + "="*70)
print("RESULTADOS FINAIS")
print("="*70 + "\n")

if results:
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    print(f"\n✓ Benchmark concluído com sucesso!")
    print(f"Total de combinações executadas: {len(results)}")
    print(f"Datasets com sucesso: {df_results['Dataset'].nunique()}")
else:
    print("Nenhum resultado foi gerado.")
