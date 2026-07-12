@echo off
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Running benchmark...
python benchmarks\benchmark.py

pause
