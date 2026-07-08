@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"

echo Instalando dependencias faltantes...
pip install -q packaging networkx

echo.
echo Executando benchmark...
python benchmark_simple.py

pause
