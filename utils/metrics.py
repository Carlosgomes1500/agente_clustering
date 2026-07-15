# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def map_clusters_to_classes(y_true, y_pred):
    """
    Como a clusterização gera rótulos arbitrários (ex: o cluster '0' pode ser a classe '2'),
    precisamos mapear os rótulos preditos para as verdadeiras classes buscando a maior interseção.
    Usamos o Algoritmo Húngaro (linear_sum_assignment) para isso.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Criar matriz de confusão/custo (tamanho: max_class x max_cluster)
    max_label = max(y_true.max(), y_pred.max()) + 1
    cost_matrix = np.zeros((max_label, max_label), dtype=int)
    
    for i in range(y_true.size):
        cost_matrix[y_true[i], y_pred[i]] += 1
        
    # linear_sum_assignment minimiza o custo, então passamos negativo para maximizar acertos
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    
    # Criar dicionário de mapeamento {cluster_predito: classe_real}
    mapping = {pred: true for true, pred in zip(row_ind, col_ind)}
    
    # Substituir os rótulos
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    
    return y_pred_mapped

def get_classification_metrics(y_true, y_pred):
    """
    Mapeia os clusters para as classes reais e calcula Accuracy, Precision, Recall e F1-Score.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0, 0.0
        
    y_pred_mapped = map_clusters_to_classes(y_true, y_pred)
    
    # Usamos average='weighted' para lidar bem com datasets cujas classes são desbalanceadas
    acc = accuracy_score(y_true, y_pred_mapped)
    prec = precision_score(y_true, y_pred_mapped, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred_mapped, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred_mapped, average='weighted', zero_division=0)
    
    return acc, prec, rec, f1
