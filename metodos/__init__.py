"""
Capa 4 - Metodos iterativos para sistemas lineales.
Todos los metodos comparten la misma firma de interfaz.
"""

from .jacobi import resolver as jacobi
from .gauss_seidel import resolver as gauss_seidel
from .sor import resolver as sor
from .ssor import resolver as ssor

__all__ = ["jacobi", "gauss_seidel", "sor", "ssor"]
