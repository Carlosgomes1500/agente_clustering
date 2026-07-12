# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_data(X, y, dataset_name="Dataset"):
    """
    Pré-processamento robusto para os dados:
    - Converte matrizes esparsas para densas.
    - Achata imagens multidimensionais (Flatten).
    - Remove NaNs.
    - Codifica colunas categóricas.
    - Normaliza os dados (StandardScaler).
    - Codifica a variável alvo y.
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
        if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
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
    Previne problemas de performance em algoritmos complexos.
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
