import subprocess
import sys

print("Instalando dependências...")
deps = ['scikit-learn', 'pandas', 'numpy', 'scipy', 'scikit-fuzzy', 'tabulate']

for dep in deps:
    print(f"  - Instalando {dep}...", end=" ")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', dep], 
                          capture_output=True)
    if result.returncode == 0:
        print("✓")
    else:
        print(f"✗ ({result.stderr.decode()})")

print("\nExecutando benchmark...\n")
subprocess.run([sys.executable, 'benchmark_clustering.py'])
