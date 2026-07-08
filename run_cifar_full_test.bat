@echo off
chcp 65001 > nul
cd /d "c:\Users\carlo\OneDrive\Documentos\agente_clustering"
echo ========================================================================
echo TESTE EXTREMO: CIFAR-10 COMPLETO (60.000 imagens)
echo ========================================================================
echo.
echo Este teste vai:
echo  - Baixar CIFAR-10 (163 MB)
echo  - Carregar 60.000 imagens
echo  - Extrair features Vectorizadas
echo  - Extrair features HoG (20-30 minutos!)
echo  - Rodar K-Means em ambos
echo  - Calcular métricas com proteção de memória
echo.
echo Tempo total estimado: 40-50 minutos
echo.
pause
python test_cifar_full.py
echo.
pause
