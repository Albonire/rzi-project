@echo off
echo ========================================
echo  RZI - Instalacion de Dependencias
echo ========================================
echo.
echo Creando entorno virtual...
python -m venv venv
if %errorlevel% neq 0 (
    echo Error: No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo Activando entorno virtual...
call venv\Scripts\activate.bat
echo Instalando dependencias...
pip install -r requirements.txt
echo.
echo ========================================
echo  Instalacion completada!
echo ========================================
echo Para ejecutar: ejecutar.bat
pause
