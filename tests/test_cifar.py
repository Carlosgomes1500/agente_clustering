#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Teste isolado para download e carregamento do CIFAR-10
Usa urllib para download, tarfile para extração e pickle para leitura
"""

import os
import sys
import urllib.request
import tarfile
import pickle
import numpy as np
from pathlib import Path

def load_cifar10(data_dir='../data/cifar_data', url='https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'):
    """
    Baixa e carrega o dataset CIFAR-10
    
    Args:
        data_dir: Diretório para armazenar os dados
        url: URL oficial do CIFAR-10
    
    Returns:
        X: numpy array (50000, 3072) - dados de treino
        y: numpy array (50000,) - rótulos de treino
    """
    
    # Criar diretório se não existir
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    tar_path = os.path.join(data_dir, 'cifar-10-python.tar.gz')
    extract_dir = os.path.join(data_dir, 'cifar-10-batches-py')
    
    # 1. DOWNLOAD
    if not os.path.exists(tar_path):
        print(f"[Download] CIFAR-10 from {url}")
        try:
            urllib.request.urlretrieve(url, tar_path, reporthook=_download_progress)
            print("\n✓ Download concluído!")
        except Exception as e:
            print(f"\n✗ Erro no download: {e}")
            raise
    else:
        print(f"[Cache] Arquivo já existe em {tar_path}")
    
    # 2. EXTRAÇÃO
    if not os.path.exists(extract_dir):
        print(f"[Extract] Extraindo para {extract_dir}")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("✓ Extração concluída!")
        except Exception as e:
            print(f"✗ Erro na extração: {e}")
            raise
    else:
        print(f"[Cache] Diretório já extraído em {extract_dir}")
    
    # 3. CARREGAMENTO DOS ARQUIVOS PICKLE
    print("[Load] Carregando arquivos pickle...")
    
    X_list = []
    y_list = []
    
    # Carregar 5 batches de treino
    for batch_num in range(1, 6):
        batch_file = os.path.join(extract_dir, f'data_batch_{batch_num}')
        print(f"  Lendo {batch_file}...")
        
        try:
            with open(batch_file, 'rb') as f:
                batch_data = pickle.load(f, encoding='bytes')
            
            X_list.append(batch_data[b'data'])
            y_list.extend(batch_data[b'labels'])
        except Exception as e:
            print(f"  ✗ Erro ao ler {batch_file}: {e}")
            raise
    
    # Concatenar todos os batches
    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    
    print("✓ Carregamento concluído!")
    
    return X, y

def _download_progress(block_num, block_size, total_size):
    """Callback para mostrar progresso do download"""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 // total_size, 100)
    sys.stdout.write(f'\r  [{percent}%] {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB')
    sys.stdout.flush()

if __name__ == "__main__":
    print("="*70)
    print("TESTE DE DOWNLOAD E CARREGAMENTO DO CIFAR-10")
    print("="*70 + "\n")
    
    try:
        X, y = load_cifar10()
        
        print("\n" + "="*70)
        print("RESULTADOS")
        print("="*70)
        print(f"\nShape de X: {X.shape}")
        print(f"Shape de y: {y.shape}")
        print(f"Dtype de X: {X.dtype}")
        print(f"Dtype de y: {y.dtype}")
        print(f"Classes únicas: {np.unique(y)}")
        print(f"Número de classes: {len(np.unique(y))}")
        print(f"Range de valores: X [{X.min()}, {X.max()}] | y [{y.min()}, {y.max()}]")
        
        print("\n✓ CIFAR-10 carregado com sucesso!")
        
    except Exception as e:
        print(f"\n✗ Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
