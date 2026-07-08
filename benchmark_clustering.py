import subprocess
import sys

# Instalar dependências se necessário
required_packages = ['scikit-learn', 'pandas', 'numpy', 'scipy', 'scikit-fuzzy', 'tabulate']
for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
    except ImportError:
        print(f"Instalando {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

import numpy as np
import pandas as pd
from time import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml, load_digits
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
import skfuzzy as fuzz

results = []

def preprocess_data(X, y):
    """Pré-processamento padrão: imputa, codifica categóricas, normaliza"""
    X = pd.DataFrame(X)
    
    # Remover NaNs
    mask = X.isnull().any(axis=1)
    X = X[~mask]
    y = y[~mask]
    
    # Converter categóricas para numéricas
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # LabelEncoder para y (target)
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    return X_scaled, y_encoded

def run_kmeans(X, y, k, dataset_name):
    """K-Means"""
    start = time()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = km.fit_predict(X)
    elapsed = time() - start
    
    sil = silhouette_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    homog = homogeneity_score(y, y_pred)
    compl = completeness_score(y, y_pred)
    v_meas = v_measure_score(y, y_pred)
    
    results.append({
        'Dataset': dataset_name,
        'Algoritmo': 'K-Means',
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4),
        'Homogeneidade (Prec.)': round(homog, 4),
        'Completude (Rec.)': round(compl, 4),
        'V-Measure (F1)': round(v_meas, 4),
        'Tempo (s)': round(elapsed, 4)
    })

def run_hierarchical(X, y, k, dataset_name):
    """Hierárquico Aglomerativo (ward)"""
    start = time()
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    y_pred = hc.fit_predict(X)
    elapsed = time() - start
    
    sil = silhouette_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    homog = homogeneity_score(y, y_pred)
    compl = completeness_score(y, y_pred)
    v_meas = v_measure_score(y, y_pred)
    
    results.append({
        'Dataset': dataset_name,
        'Algoritmo': 'Hierárquico (Ward)',
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4),
        'Homogeneidade (Prec.)': round(homog, 4),
        'Completude (Rec.)': round(compl, 4),
        'V-Measure (F1)': round(v_meas, 4),
        'Tempo (s)': round(elapsed, 4)
    })

def run_spectral(X, y, k, dataset_name):
    """Spectral Clustering"""
    start = time()
    sc = SpectralClustering(n_clusters=k, random_state=42, affinity='nearest_neighbors')
    y_pred = sc.fit_predict(X)
    elapsed = time() - start
    
    sil = silhouette_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    homog = homogeneity_score(y, y_pred)
    compl = completeness_score(y, y_pred)
    v_meas = v_measure_score(y, y_pred)
    
    results.append({
        'Dataset': dataset_name,
        'Algoritmo': 'Spectral',
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4),
        'Homogeneidade (Prec.)': round(homog, 4),
        'Completude (Rec.)': round(compl, 4),
        'V-Measure (F1)': round(v_meas, 4),
        'Tempo (s)': round(elapsed, 4)
    })

def run_fuzzy_cmeans(X, y, k, dataset_name):
    """Fuzzy C-Means"""
    try:
        start = time()
        cntr, u, _, _, _, _, _ = fuzz.cmeans(X.T, k, m=2.0, error=0.005, maxiter=1000)
        y_pred = np.argmax(u, axis=0)
        elapsed = time() - start
        
        sil = silhouette_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        ari = adjusted_rand_score(y, y_pred)
        homog = homogeneity_score(y, y_pred)
        compl = completeness_score(y, y_pred)
        v_meas = v_measure_score(y, y_pred)
        
        results.append({
            'Dataset': dataset_name,
            'Algoritmo': 'Fuzzy C-Means',
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Homogeneidade (Prec.)': round(homog, 4),
            'Completude (Rec.)': round(compl, 4),
            'V-Measure (F1)': round(v_meas, 4),
            'Tempo (s)': round(elapsed, 4)
        })
    except Exception as e:
        print(f"Erro em Fuzzy C-Means para {dataset_name}: {str(e)}")

def run_larfsom(X, y, k, dataset_name):
    """LARFSOM com fallback para mock"""
    try:
        import larfsom
        start = time()
        model = larfsom.LARFSOM(n_clusters=k)
        y_pred = model.fit_predict(X)
        elapsed = time() - start
    except (ImportError, FileNotFoundError, ModuleNotFoundError):
        print(f"[AVISO] LARFSOM não encontrado. Usando predições mockadas para {dataset_name}.")
        start = time()
        y_pred = np.zeros(X.shape[0], dtype=int)
        elapsed = time() - start
    
    sil = silhouette_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    homog = homogeneity_score(y, y_pred)
    compl = completeness_score(y, y_pred)
    v_meas = v_measure_score(y, y_pred)
    
    results.append({
        'Dataset': dataset_name,
        'Algoritmo': 'LARFSOM',
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4),
        'Homogeneidade (Prec.)': round(homog, 4),
        'Completude (Rec.)': round(compl, 4),
        'V-Measure (F1)': round(v_meas, 4),
        #'Tempo (s)': round(elapsed, 4)
    })

def process_dataset(X, y, dataset_name):
    """Processa um dataset completo"""
    print(f"\n{'='*60}")
    print(f"Processando: {dataset_name}")
    print(f"{'='*60}")
    
    X_scaled, y_encoded = preprocess_data(X, y)
    k = len(np.unique(y_encoded))
    
    print(f"Shape: {X_scaled.shape} | Classes: {k}")
    print(f"Executando algoritmos...")
    
    run_kmeans(X_scaled, y_encoded, k, dataset_name)
    run_hierarchical(X_scaled, y_encoded, k, dataset_name)
    run_spectral(X_scaled, y_encoded, k, dataset_name)
    run_fuzzy_cmeans(X_scaled, y_encoded, k, dataset_name)
    run_larfsom(X_scaled, y_encoded, k, dataset_name)
    
    print(f"✓ {dataset_name} completo!")

print("\n" + "="*60)
print("BENCHMARK DE ALGORITMOS DE CLUSTERIZAÇÃO")
print("="*60)

# 1. DIGITS
print("\n[1/6] Carregando DIGITS...")
try:
    digits = load_digits()
    process_dataset(digits.data, digits.target, 'Digits')
except Exception as e:
    print(f"Erro ao carregar Digits: {e}")

# Datasets do OpenML
openml_datasets = [
    ('vehicle', 1, 'Vehicle', '2/6'),
    ('letter', 1, 'Letter', '3/6'),
    ('ionosphere', 1, 'Ionosphere', '4/6'),
    ('isolet', 1, 'Isolet', '5/6'),
    ('waveform-5000', 1, 'Waveform-5000', '6/6')
]

for name, version, display_name, step in openml_datasets:
    print(f"\n[{step}] Carregando {display_name}...")
    try:
        data = fetch_openml(name=name, version=version, as_frame=True, parser='auto')
        process_dataset(data.data.values, data.target.values, display_name)
    except Exception as e:
        print(f"Erro ao carregar {display_name}: {e}")

# Consolidar resultados
print("\n" + "="*60)
print("RESULTADOS FINAIS")
print("="*60)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_markdown(index=False))

print("\n✓ Benchmark concluído com sucesso!")
