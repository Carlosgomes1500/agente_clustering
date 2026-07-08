@echo off
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"

echo Installing dependencies...
pip install -q scikit-learn pandas numpy scipy scikit-fuzzy tabulate

echo.
echo Running benchmark...
python benchmark_clustering.py

pause
