#!/bin/bash
echo "========================================"
echo " RZI - Instalacion de Dependencias"
echo "========================================"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
echo "Instalacion completada!"
echo "Para ejecutar: ./ejecutar.sh"
