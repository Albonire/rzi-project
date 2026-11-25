"""
Diagnostico del entorno y de cada herramienta de RZI, sin interfaz grafica.

Uso (desde la raiz del proyecto):
    python scripts/diagnostico.py

Si alguna herramienta de la aplicacion "no hace nada" o deja una pestana
vacia, ejecute esto y comparta el informe que imprime.
"""

import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import matplotlib
matplotlib.use('Agg')

from services.diagnostico import ejecutar_diagnostico


def main():
    texto, ok = ejecutar_diagnostico(
        progreso=lambda f, t: print(f'  ... {t}', flush=True) if f < 1 else None)
    print()
    print(texto)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
