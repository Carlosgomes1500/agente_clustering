# -*- coding: utf-8 -*-
from time import time
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from utils.data_processing import subsample_if_needed

def evaluate_algorithm(wrapper_class, alg_name, X, y, k, dataset_name, needs_subsample=False):
    """
    Avalia um algoritmo de clusterização.
    Retorna um dicionário com os resultados ou None se houver erro.
    """
    try:
        if needs_subsample:
            X_eval, y_eval = subsample_if_needed(X, y)
        else:
            X_eval, y_eval = X, y
            
        start = time()
        model = wrapper_class(n_clusters=k)
        y_pred = model.fit_predict(X_eval)
        elapsed = time() - start
        
        sil = silhouette_score(X_eval, y_pred)
        db = davies_bouldin_score(X_eval, y_pred)
        ari = adjusted_rand_score(y_eval, y_pred)
        
        print(f"    ✓ {alg_name}: {round(elapsed, 2)}s")
        return {
            'Dataset': dataset_name,
            'Algoritmo': alg_name,
            'Silhouette': round(sil, 4),
            'Davies-Bouldin': round(db, 4),
            'ARI': round(ari, 4),
            'Tempo (s)': round(elapsed, 4)
        }
    except Exception as e:
        print(f"    ✗ {alg_name}: {e}")
        return None
