@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"
echo Iniciando teste de extracção de características CIFAR-10...
echo.
python test_cifar_features.py
echo.
pause
