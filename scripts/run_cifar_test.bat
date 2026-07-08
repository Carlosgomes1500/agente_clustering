@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"
echo Iniciando teste do CIFAR-10...
echo.
python test_cifar.py
echo.
echo.
pause
