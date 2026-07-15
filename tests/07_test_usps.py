#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import fetch_openml
from benchmarks.main import process_single_dataset
from benchmarks.results import compile_results

if __name__ == "__main__":
    print("Carregando dataset USPS (OpenML)...")
    dataset = fetch_openml(name='usps', version=1, as_frame='auto', parser='auto')
    resultados = process_single_dataset(dataset.data.values, dataset.target.values, 'USPS', fast_mode=True)
    compile_results(resultados)
