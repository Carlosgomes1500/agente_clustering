@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"
python benchmarks\benchmark.py --mode fast
pause
