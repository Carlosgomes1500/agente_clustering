#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste Extremo de Extração de Características - CIFAR-10 Completo (60.000 imagens)
Compara Raw Pixels vs HoG com prevenção de Out-of-Memory
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
import gc

print("  ✓ Todas as dependências carregadas\n")

# 2. FUNÇÃO LOAD_CIFAR10_COMPLETE (COM TEST_BATCH)
def load_cifar10_complete(data_dir='./cifar_data', url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    """
    Baixa e carrega o dataset CIFAR-10 COMPLETO (60.000 imagens)
    Inclui 5 batches de treino + 1 batch de teste
    """
    
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
    
    # Carregamento COMPLETO (5 treino + 1 teste = 60.000)
    print("[Load] Carregando todos os 6 batches (treino + teste)...")
    
    X_list = []
    y_list = []
    
    # Carregar 5 batches de treino
    for batch_num in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{batch_num}')
        print(f"  Lendo data_batch_{batch_num}...")
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            X_list.append(batch_data[b'data'])
            y_list.extend(batch_data[b'labels'])
        except Exception as e:
            print(f"  ✗ Erro ao ler batch {batch_num}: {e}")
            raise
    
    # Carregar batch de teste
    test_batch_file = os.path.join(extract_dir, 'test_batch')
    print(f"  Lendo test_batch...")
    try:
        with open(test_batch_file, 'rb') as f:
            test_data = pickle.load(f, encoding='bytes')
        X_list.append(test_data[b'data'])
        y_list.extend(test_data[b'labels'])
    except Exception as e:
        print(f"  ✗ Erro ao ler test_batch: {e}")
        raise
    
    # Concatenar todos os dados (60.000 imagens)
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    print(f"✓ Carregamento completo!")
    print(f"  Total: X.shape={X.shape}, y.shape={y.shape}\n")
    
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
    print(f"  Input shape: {X.shape}")
    
    scaler = StandardScaler()
    X_vect = scaler.fit_transform(X)
    
    print(f"  Output shape: {X_vect.shape}\n")
    gc.collect()
    
    return X_vect

# 4. EXTRATOR 2: HOG (HISTOGRAM OF ORIENTED GRADIENTS)
def extract_hog_features(X):
    """
    Extrai features HoG do CIFAR-10 COMPLETO (60.000 imagens)
    Input: X com shape (60000, 3072) achatado em ordem RGB
    Imprime progresso a cada 10.000 imagens
    """
    print("[Feature] Extracting HoG features (60.000 imagens)...")
    print(f"  Input shape: {X.shape}")
    print(f"  ⚠️  Isto pode levar 20-30 minutos...\n")
    
    n_samples = X.shape[0]
    hog_features = []
    
    start_time = time()
    
    for i in range(n_samples):
        # Imprimir progresso a cada 10.000 imagens
        if (i + 1) % 10000 == 0:
            elapsed = time() - start_time
            eta = (elapsed / (i + 1)) * (n_samples - i - 1)
            print(f"    ⏳ Processadas {i+1}/{n_samples} imagens ({elapsed:.1f}s decorridos, ETA: {eta:.1f}s)")
        
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
    
    total_time = time() - start_time
    print(f"  ✓ HoG extraction concluído em {total_time:.1f}s")
    
    X_hog = np.array(hog_features)
    print(f"  Shape antes de scaler: {X_hog.shape}")
    
    # Aplicar StandardScaler
    scaler = StandardScaler()
    X_hog = scaler.fit_transform(X_hog)
    print(f"  Output shape: {X_hog.shape}\n")
    
    gc.collect()
    
    return X_hog

# 5. EXECUTAR K-MEANS COM PROTEÇÃO DE MEMÓRIA
def evaluate_features(X, y, feature_name, k=10):
    """
    Executa K-Means e calcula métricas
    REGRA CRÍTICA: silhouette_score DEVE usar sample_size=5000 para evitar OOM
    """
    print(f"[Clustering] {feature_name} com K-Means (k={k})...")
    
    # K-Means
    start = time()
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    y_pred = kmeans.fit_predict(X)
    elapsed = time() - start
    
    print(f"  ✓ K-Means concluído em {elapsed:.2f}s")
    
    # Silhouette Score COM PROTEÇÃO DE MEMÓRIA
    print(f"  Calculando Silhouette (sample_size=5000)...")
    start = time()
    sil = silhouette_score(X, y_pred, sample_size=5000)  # OBRIGATÓRIO: evita OOM
    sil_time = time() - start
    print(f"    ✓ {sil_time:.2f}s | Score: {sil:.4f}")
    
    # Davies-Bouldin (array completo)
    print(f"  Calculando Davies-Bouldin (array completo)...")
    start = time()
    db = davies_bouldin_score(X, y_pred)
    db_time = time() - start
    print(f"    ✓ {db_time:.2f}s | Score: {db:.4f}")
    
    # ARI (array completo)
    print(f"  Calculando ARI (array completo)...")
    start = time()
    ari = adjusted_rand_score(y, y_pred)
    ari_time = time() - start
    print(f"    ✓ {ari_time:.2f}s | Score: {ari:.4f}\n")
    
    gc.collect()
    
    return {
        'Extrator': feature_name,
        'Silhouette (Sampled)': round(sil, 4),
        'Davies-Bouldin': round(db, 4),
        'ARI': round(ari, 4)
    }

# MAIN EXECUTION
if __name__ == "__main__":
    print("="*80)
    print("TESTE EXTREMO: CIFAR-10 COMPLETO (60.000 imagens)")
    print("="*80 + "\n")
    
    try:
        # Carregar dados COMPLETOS
        print("[Download] Carregando CIFAR-10 Completo...")
        X, y = load_cifar10_complete()
        
        print(f"✓ Dados carregados: X.shape={X.shape}, y.shape={y.shape}\n")
        
        # Extrair features
        X_vect = extract_vectorized_features(X)
        X_hog = extract_hog_features(X)
        
        # Avaliar ambos
        print("\n" + "="*80)
        print("[Evaluation] Executando K-Means em ambos os extractores")
        print("="*80 + "\n")
        
        results = []
        results.append(evaluate_features(X_vect, y, 'Raw Pixels', k=10))
        results.append(evaluate_features(X_hog, y, 'HoG Features', k=10))
        
        # Tabela comparativa
        print("="*80)
        print("RESULTADOS COMPARATIVOS (60.000 imagens)")
        print("="*80 + "\n")
        
        df_results = pd.DataFrame(results)
        print(df_results.to_markdown(index=False))
        
        print("\n✓ Teste extremo concluído com sucesso!")
        print(f"  Total de amostras processadas: 60.000")
        print(f"  Extractores: 2 (Raw Pixels + HoG)")
        print(f"  Algoritmos: K-Means")
        
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
