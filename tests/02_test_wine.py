#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.datasets import load_wine
from benchmarks.main import process_single_dataset

if __name__ == "__main__":
    print("Carregando dataset Wine...")
    dataset = load_wine()
    process_single_dataset(dataset.data, dataset.target, 'Wine', fast_mode=False)
