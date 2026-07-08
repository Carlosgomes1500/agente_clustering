@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"
echo ========================================================================
echo BENCHMARK DEFINITIVO - 5 ALGORITMOS DE CLUSTERIZAÇÃO
echo ========================================================================
echo.
echo Datasets: Iris, Wine, Digits, Vehicle, Letter, Ionosphere, USPS, Isolet, Waveform-5000
echo Algoritmos: K-Means, Hierárquico (Ward), Spectral, Fuzzy C-Means, LARFSOM
echo.
echo Este teste vai executar 45 combinações (9 datasets x 5 algoritmos)
echo Tempo estimado: 15-30 minutos
echo.
pause
python benchmark_simple.py
echo.
pause
