#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import fetch_openml
from benchmarks.main import process_single_dataset

if __name__ == "__main__":
    print("Carregando dataset Letter (OpenML)...")
    dataset = fetch_openml(name='letter', version=1, as_frame='auto', parser='auto')
    process_single_dataset(dataset.data.values, dataset.target.values, 'Letter', fast_mode=False)
