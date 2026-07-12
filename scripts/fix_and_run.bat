@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"

echo Instalando dependencias faltantes...
pip install -r requirements.txt

echo.
echo Executando benchmark...
python benchmarks\benchmark.py

pause
