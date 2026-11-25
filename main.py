#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RZI - Reconstructor de Zoom Iterativo
Punto de entrada de la aplicacion.

Arquitectura: 3 capas
  Capa 1  ui/main_window.py   Interfaz grafica (PySide6 / Qt)
  Capa 2  core/               Nucleo matematico (Poisson + SOR Red-Black + Numba)
  Capa 3  services/           Metricas, exportacion, graficas y diagnostico

Ejecutar:
    python main.py
Diagnostico sin interfaz (si alguna herramienta no responde):
    python scripts/diagnostico.py
"""

import sys
import os
import traceback

# Asegurar que el directorio del proyecto este en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui     import QFont
from ui.main_window    import VentanaPrincipal, ESTILO, mostrar_error, registrar_error


def _gancho_excepciones(tipo, valor, tb):
    """
    Cualquier excepcion no atrapada (por ejemplo, dentro de un slot de Qt)
    se muestra en un dialogo y se guarda en rzi_errores.log, en vez de
    perderse en la consola.
    """
    detalle = ''.join(traceback.format_exception(tipo, valor, tb))
    sys.__stderr__.write(detalle)
    if QApplication.instance() is not None:
        mostrar_error(None, 'Error inesperado', f'{tipo.__name__}: {valor}', detalle)
    else:
        registrar_error(detalle)


def main():
    sys.excepthook = _gancho_excepciones
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(ESTILO)
    app.setFont(QFont('Segoe UI', 10))

    ventana = VentanaPrincipal()
    ventana.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
