# -*- coding: utf-8 -*-
from sklearn.datasets import load_iris, load_wine, load_digits, fetch_openml

def get_datasets(run_mode='all'):
    """
    Gera (yield) os datasets baseados no modo de execução.
    Yields:
        (X, y, dataset_name, is_fast_mode_for_this)
    """
    fast_mode = (run_mode == 'fast')

    # FASE 1: Datasets Nativos
    if run_mode in ['all', 'fast', 'native']:
        print("\n[FASE 1] Datasets nativos (sklearn.datasets)\n")

        print("[Load] Carregando Iris...")
        try:
            iris = load_iris()
            yield iris.data, iris.target, 'Iris'
        except Exception as e:
            print(f"✗ Erro ao carregar Iris: {e}")

        print("\n[Load] Carregando Wine...")
        try:
            wine = load_wine()
            yield wine.data, wine.target, 'Wine'
        except Exception as e:
            print(f"✗ Erro ao carregar Wine: {e}")

        print("\n[Load] Carregando Digits...")
        try:
            digits = load_digits()
            yield digits.data, digits.target, 'Digits'
        except Exception as e:
            print(f"✗ Erro ao carregar Digits: {e}")

    # FASE 2: Datasets OpenML
    if run_mode in ['all', 'fast', 'openml']:
        print("\n" + "="*70)
        print("[FASE 2] Datasets OpenML\n")

        openml_datasets = [
            ('vehicle', 1, 'Vehicle'),
            ('letter', 1, 'Letter'),
            ('ionosphere', 1, 'Ionosphere'),
            ('usps', 1, 'USPS'),
        ]
        
        if fast_mode:
            openml_datasets = openml_datasets[:2]

        for dataset_name, version, display_name in openml_datasets:
            print(f"\n[Load] Carregando {display_name}...")
            try:
                data = fetch_openml(name=dataset_name, version=version, as_frame='auto', parser='auto')
                yield data.data.values, data.target.values, display_name
            except Exception as e:
                print(f"✗ Aviso ao carregar {display_name}: {str(e)[:100]}")

        if not fast_mode:
            print(f"\n[Load] Carregando Isolet...")
            try:
                isolet = fetch_openml(name='isolet', version=1, as_frame='auto', parser='auto')
                yield isolet.data.values, isolet.target.values, 'Isolet'
            except Exception as e:
                print(f"✗ Aviso ao carregar Isolet: {str(e)[:100]}")

            print(f"\n[Load] Carregando Waveform-5000...")
            try:
                waveform = fetch_openml(name='waveform-5000', version=1, as_frame='auto', parser='auto')
                yield waveform.data.values, waveform.target.values, 'Waveform-5000'
            except Exception as e:
                print(f"✗ Aviso ao carregar Waveform-5000: {str(e)[:100]}")
