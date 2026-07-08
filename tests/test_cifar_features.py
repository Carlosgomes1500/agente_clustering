#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste de Extração de Características no CIFAR-10
Compara Raw Pixels vs HoG (Histogram of Oriented Gradients)
"""

import os
import sys
import urllib.request
import tarfile
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from time import time

# 1. INSTALAR SCIKIT-IMAGE SE NECESSÁRIO
print("[Setup] Verificando dependências...")
try:
    from skimage.feature import hog
    print("  ✓ scikit-image já instalado")
except ImportError:
    print("  ✗ scikit-image não encontrado. Instalando...")
    os.system("pip install -q scikit-image")
    from skimage.feature import hog
    print("  ✓ scikit-image instalado")

# Importar sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

print("  ✓ Todas as dependências carregadas\n")

# 2. FUNÇÃO LOAD_CIFAR10 (REUTILIZADA)
def load_cifar10(data_dir='../data/cifar_data', url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    """Baixa e carrega o dataset CIFAR-10"""
    
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    extract_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    # Download
    if not os.path.exists(tar_path):
        print(f"[Download] CIFAR-10 from {url}")
        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=_download_progress)
            print("\n✓ Download concluído!")
        except Exception as e:
            print(f"\n✗ Erro no download: {e}")
            raise
    else:
        print(f"[Cache] Arquivo já existe")
    
    # Extração
    if not os.path.exists(extract_dir):
        print(f"[Extract] Extraindo...")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("✓ Extração concluída!")
        except Exception as e:
            print(f"✗ Erro na extração: {e}")
            raise
    else:
        print(f"[Cache] Já extraído")
    
    # Carregamento
    print("[Load] Carregando pickle files...")
    
    X_list = []
    y_list = []
    
    for batch_num in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{batch_num}')
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            X_list.append(batch_data[b'data'])
            y_list.extend(batch_data[b'labels'])
        except Exception as e:
            print(f"✗ Erro ao ler batch {batch_num}: {e}")
            raise
    
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    print("✓ Carregamento concluído!\n")
    
    return X, y

def _download_progress(block_num, block_size, total_size):
    """Callback para mostrar progresso do download"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 // total_size, 100)
    sys.stdout.write(f'\r  [{percent}%] {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB')
    sys.stdout.flush()

# 3. EXTRATOR 1: RAW PIXELS (VETORIZADO)
def extract_vectorized_features(X):
    """Aplica StandardScaler aos raw pixels"""
    print("[Feature] Extracting vectorized features (raw pixels)...")
    scaler = StandardScaler()
    X_vect = scaler.fit_transform(X)
    print(f"  Shape: {X_vect.shape}\n")
    return X_vect

# 4. EXTRATOR 2: HOG (HISTOGRAM OF ORIENTED GRADIENTS)
def extract_hog_features(X):
    """
    Extrai features HoG do CIFAR-10
    Input: X com shape (n_samples, 3072) achatado em ordem RGB
    """
    print("[Feature] Extracting HoG features...")
    print(f"  Input shape: {X.shape}")
    
    n_samples = X.shape[0]
    hog_features = []
    
    for i in range(n_samples):
        if (i + 1) % 500 == 0:
            print(f"    Processadas {i+1}/{n_samples} imagens...")
        
        # Reconstruir imagem: CIFAR vem em ordem RGB achatada
        img_flat = X[i].reshape(3, 32, 32)  # (3, 32, 32)
        img = img_flat.transpose(1, 2, 0)   # (32, 32, 3)
        
        # Normalizar para [0, 1]
        img = img / 255.0 if img.max() > 1 else img
        
        # Computar HoG
        hog_feat = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            channel_axis=-1
        )
        
        hog_features.append(hog_feat)
    
    X_hog = np.array(hog_features)
    print(f"  Shape antes de scaler: {X_hog.shape}")
    
    # Aplicar StandardScaler
    scaler = StandardScaler()
    X_hog = scaler.fit_transform(X_hog)
    print(f"  Shape após scaler: {X_hog.shape}\n")
    
    return X_hog

# 5. EXECUTAR K-MEANS E CALCULAR MÉTRICAS
def evaluate_features(X, y, feature_name, k=10):
    """Executa K-Means e calcula métricas"""
    print(f"[Clustering] {feature_name} com K-Means (k={k})...")
    
    start = time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    elapsed = time() - start
    
    sil = silhouette_score(X, y_pred)
    db = davies_bouldin_score(X, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    
    print(f"  ✓ Tempo: {elapsed:.2f}s")
    print(f"  Silhouette: {sil:.4f}")
    print(f"  Davies-Bouldin: {db:.4f}")
    print(f"  ARI: {ari:.4f}\n")
    
    return {
        'Extrator': feature_name,
        'Silhouette': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4)
    }

# MAIN EXECUTION
if __name__ == "__main__":
    print("="*70)
    print("TESTE DE EXTRAÇÃO DE CARACTERÍSTICAS - CIFAR-10")
    print("="*70 + "\n")
    
    try:
        # Carregar dados
        print("[Download] Carregando CIFAR-10...")
        X, y = load_cifar10()
        
        # REGRA CRÍTICA: Slice para 3000 amostras
        print("[Slice] Aplicando slice: X[:3000], y[:3000]")
        X = X[:3000]
        y = y[:3000]
        print(f"  Shape: X={X.shape}, y={y.shape}\n")
        
        # Extrair features
        X_vect = extract_vectorized_features(X)
        X_hog = extract_hog_features(X)
        
        # Avaliar ambos
        results = []
        results.append(evaluate_features(X_vect, y, 'Raw Pixels', k=10))
        results.append(evaluate_features(X_hog, y, 'HoG Features', k=10))
        
        # Tabela comparativa
        print("="*70)
        print("RESULTADOS COMPARATIVOS")
        print("="*70 + "\n")
        
        df_results = pd.DataFrame(results)
        print(df_results.to_markdown(index=False))
        
        print("\n✓ Teste concluído com sucesso!")
        
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
