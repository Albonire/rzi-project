#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RZI - Reconstructor de Zoom Iterativo
Punto de entrada de la aplicacion.

Arquitectura: 3 capas
  Capa 1  ui/main_window.py   Interfaz grafica (PySide6 / Qt)
  Capa 2  core/               Nucleo matematico (Poisson + SOR Red-Black + Numba)
  Capa 3  services/           Metricas, exportacion y graficas

Ejecutar:
    python main.py
"""

import sys
import os

# Asegurar que el directorio del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui     import QFont
from ui.main_window    import VentanaPrincipal, ESTILO


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(ESTILO)
    app.setFont(QFont('Segoe UI', 10))

    ventana = VentanaPrincipal()
    ventana.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
