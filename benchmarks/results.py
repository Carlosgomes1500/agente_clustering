# -*- coding: utf-8 -*-
import pandas as pd

def compile_results(results_list, output_file='resultados_benchmark.csv'):
    """
    Recebe a lista de resultados gerada pelos algoritmos, imprime na tela e salva em CSV.
    """
    print("\n" + "="*70)
    print("RESULTADOS FINAIS")
    print("="*70 + "\n")

    if results_list:
        df_results = pd.DataFrame(results_list)
        print(df_results.to_markdown(index=False))
        
        # Salvar em CSV
        df_results.to_csv(output_file, index=False)
        
        print(f"\n✓ Benchmark concluído com sucesso!")
        print(f"Resultados salvos em: {output_file}")
        print(f"Total de execuções: {len(results_list)}")
        print(f"Datasets avaliados: {df_results['Dataset'].nunique()}")
    else:
        print("Nenhum resultado foi gerado.")
