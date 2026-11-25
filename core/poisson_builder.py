"""
Capa 2 - Construccion de los problemas de Poisson que resuelve el solver SOR.

Este modulo prepara dos tipos de problema:

1. Problema clasico con mascara (Enfoque B, uso didactico):
   se marca 1 de cada factor x factor pixeles como "conocido" (condicion de
   Dirichlet) y se resuelve, para cada pixel desconocido (i, j),

       4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = b[i,j]

   El lado izquierdo es (-Laplaciano) discreto de 5 puntos. Por eso, para que
   la solucion "siga" a una imagen guia G, el termino independiente debe ser
   b = -Laplaciano(G). Con b = 0 se obtiene la interpolacion armonica (Laplace).
   Este problema se usa en el Barrido de Omega y en el Benchmark.

2. Problema de gradiente "screened" (modelo RZI de upscaling):
   en una malla centrada en pixeles (salida exacta h*factor x w*factor) se
   resuelve

       (4 + lam) * u[i,j] - (suma de los 4 vecinos) = -div(v)[i,j] + lam * t[i,j]

   que es la ecuacion de Euler-Lagrange de la energia
       E(u) = lam * ||u - t||^2 + ||grad(u) - v||^2
   donde t es la estimacion actual (corregida por back-projection) y v es su
   campo de gradientes amplificado en los bordes. Ver core/upscaler.py.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt


# Utilidades generales

def imagen_a_grises(imagen: np.ndarray) -> np.ndarray:
    """
    Convierte cualquier imagen a escala de grises float64 en [0, 1].
    Usa la formula de luminancia ITU-R BT.601 (la misma que la Y de YCbCr).
    """
    img = np.asarray(imagen, dtype=np.float64)
    if img.max() > 1.01:
        img = img / 255.0
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            return (0.299 * img[:, :, 0]
                    + 0.587 * img[:, :, 1]
                    + 0.114 * img[:, :, 2])
        return img[:, :, 0]
    raise ValueError(f"Formato de imagen no soportado: {imagen.shape}")


def gradiente_adelante(G: np.ndarray) -> tuple:
    """
    Gradiente discreto con diferencias hacia adelante:
        gx[i,j] = G[i,j+1] - G[i,j]     gy[i,j] = G[i+1,j] - G[i,j]
    La ultima columna (gx) y la ultima fila (gy) quedan en cero.
    """
    gx = np.zeros_like(G)
    gy = np.zeros_like(G)
    gx[:, :-1] = G[:, 1:] - G[:, :-1]
    gy[:-1, :] = G[1:, :] - G[:-1, :]
    return gx, gy


def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """
    Divergencia discreta de un campo vectorial (gx, gy), con diferencias
    hacia atras: div[i,j] = (gx[i,j]-gx[i,j-1]) + (gy[i,j]-gy[i-1,j]).

    Combinada con gradiente_adelante() reproduce exactamente el Laplaciano
    de 5 puntos en el interior de la malla:
        divergence(*gradiente_adelante(G))[i,j]
            == G[i,j+1] + G[i,j-1] + G[i+1,j] + G[i-1,j] - 4*G[i,j]
    """
    gx_prev = np.zeros_like(gx)
    gy_prev = np.zeros_like(gy)
    gx_prev[:, 1:] = gx[:, :-1]
    gy_prev[1:, :] = gy[:-1, :]
    return (gx - gx_prev) + (gy - gy_prev)


def amplified_guidance(G: np.ndarray, beta: float = 1.6, tau: float = 0.05,
                        sigma_noise: float = 0.6) -> tuple:
    """
    Campo de gradientes amplificado v = s(|grad G|) * grad G.

        s(t) = 1 + (beta - 1) * t / (t + tau)

    Si el gradiente es grande (borde real, t >> tau), s(t) se acerca a beta y
    el borde se refuerza. Si es pequeno (ruido o zona plana, t << tau),
    s(t) se acerca a 1 y no se amplifica nada.

    Antes de derivar se suaviza G con un Gaussiano (sigma_noise) para no
    amplificar ruido de alta frecuencia ni el "aliasing" de la imagen
    pequena. Las derivadas son hacia adelante (ver divergence()).
    """
    Gs = gaussian_filter(G, sigma=sigma_noise) if sigma_noise > 0 else G
    gx, gy = gradiente_adelante(Gs)
    mag = np.sqrt(gx * gx + gy * gy)
    s = 1.0 + (beta - 1.0) * mag / (mag + tau)
    return s * gx, s * gy


# Parametro de relajacion omega

def omega_young(rho_jacobi: float) -> float:
    """
    Formula de Young para el omega optimo de SOR, valida para matrices
    "consistentemente ordenadas" (el estencil de 5 puntos con ordenamiento
    Rojo-Negro lo es):
        omega_opt = 2 / (1 + sqrt(1 - rho_J^2))
    donde rho_J es el radio espectral de la iteracion de Jacobi.
    """
    rho = min(max(float(rho_jacobi), 0.0), 0.999999)
    return 2.0 / (1.0 + np.sqrt(1.0 - rho * rho))


def omega_opt_rectangular(nrows: int, ncols: int, lam: float = 0.0) -> float:
    """
    Omega optimo para una malla rectangular de nrows x ncols puntos con
    frontera fija (las filas/columnas de borde no se actualizan), sin
    pixeles fijos en el interior.

    El radio espectral de Jacobi del operador (4 + lam) I - (vecinos) es
        rho_J = (2*cos(pi/(m+1)) + 2*cos(pi/(n+1))) / (4 + lam)
    con m = nrows - 2 y n = ncols - 2 puntos interiores.
    Con lam = 0 y malla cuadrada se reduce a la formula clasica
        omega_opt = 2 / (1 + sin(pi/(n+1))).
    """
    m = max(1, nrows - 2)
    n = max(1, ncols - 2)
    rho = (2.0 * np.cos(np.pi / (m + 1)) + 2.0 * np.cos(np.pi / (n + 1))) / (4.0 + lam)
    return omega_young(rho)


def calcular_omega_optimo(nrows: int, ncols: int) -> float:
    """Omega optimo de la malla completa (sin mascara interior), lam = 0."""
    return omega_opt_rectangular(nrows, ncols, 0.0)


def estimar_rho_jacobi(mascara: np.ndarray, iters: int = 300,
                        semilla: int = 0) -> float:
    """
    Estima el radio espectral de la iteracion de Jacobi del problema con
    mascara, por el metodo de la potencia.

    Por que hace falta: la formula de malla completa supone que todos los
    pixeles interiores son incognitas. En el problema con mascara, cada
    'factor' pixeles hay uno fijo (Dirichlet), lo que divide la malla en
    celdas pequenas y reduce mucho el radio espectral. Resultado: el omega
    optimo real es bastante menor que el de la formula de malla completa.

    Metodo de la potencia: se aplica repetidamente el operador de Jacobi
        (J x)[i,j] = (x[i-1,j] + x[i+1,j] + x[i,j-1] + x[i,j+1]) / 4
    (solo en pixeles desconocidos) a un vector aleatorio. El cociente de
    normas converge a |lambda_max| = rho_J. Se usa J^2 porque los
    autovalores de Jacobi aparecen en pares +rho y -rho.
    """
    desconocido = ~np.asarray(mascara, dtype=bool)
    desconocido[0, :] = desconocido[-1, :] = False
    desconocido[:, 0] = desconocido[:, -1] = False
    if not desconocido.any():
        return 0.0

    rng = np.random.default_rng(semilla)
    x = rng.random(desconocido.shape) * desconocido

    def aplicar_j(v):
        y = np.zeros_like(v)
        y[1:-1, 1:-1] = 0.25 * (v[:-2, 1:-1] + v[2:, 1:-1]
                                + v[1:-1, :-2] + v[1:-1, 2:])
        return y * desconocido

    rho = 0.0
    for _ in range(max(2, iters // 2)):
        nx = np.linalg.norm(x)
        if nx == 0.0:
            return 0.0
        x = x / nx
        y = aplicar_j(aplicar_j(x))
        rho = np.sqrt(np.linalg.norm(y))
        x = y
    return float(rho)


def estimar_omega_optimo(mascara: np.ndarray, iters: int = 300) -> float:
    """Omega optimo del problema con mascara: Young con rho_J estimado."""
    return omega_young(estimar_rho_jacobi(mascara, iters))


# Problema 1: clasico con mascara (didactico)

def construir_problema(imagen_gris: np.ndarray, factor: int = 4) -> dict:
    """
    Prepara el problema de Poisson clasico a partir de una imagen (Enfoque B).

    1. Mascara: 1 de cada factor x factor pixeles es "conocido" (Dirichlet),
       mas los bordes de la imagen.
    2. Guia: los pixeles desconocidos se rellenan con el conocido mas cercano
       (solo se usa informacion conocida, no la verdad de campo) y se suaviza.
    3. b = -Laplaciano(guia) con el estencil de 5 puntos. El signo menos es
       necesario porque el solver resuelve (-Laplaciano) u = b.

    Retorna un dict con:
        'imagen_original', 'imagen_baja' (conocidos, resto 0), 'mascara',
        'b', 'forma', 'omega_opt' (estimado con la mascara),
        'omega_malla_completa' (formula sin mascara, solo referencia),
        'factor'.
    """
    if imagen_gris.ndim != 2:
        raise ValueError("Se espera imagen 2D en escala de grises.")

    nrows, ncols = imagen_gris.shape

    mascara = np.zeros((nrows, ncols), dtype=np.bool_)
    mascara[::factor, ::factor] = True
    mascara[0, :] = True
    mascara[-1, :] = True
    mascara[:, 0] = True
    mascara[:, -1] = True

    imagen_baja = np.zeros((nrows, ncols), dtype=np.float64)
    imagen_baja[mascara] = imagen_gris[mascara]

    fill_idx = distance_transform_edt(~mascara, return_distances=False,
                                      return_indices=True)
    guia = gaussian_filter(imagen_gris[fill_idx[0], fill_idx[1]], sigma=1.0)

    # b = -Laplaciano(guia): con este signo, (-Lap) u = -Lap(guia) => u sigue
    # a la guia. (Con el signo contrario la solucion se aleja de la guia y
    # resulta peor que Laplace puro; verificado en tests/test_rzi.py.)
    b = -divergence(*gradiente_adelante(guia))
    b[mascara] = 0.0

    return {
        'imagen_original':      imagen_gris.astype(np.float64),
        'imagen_baja':          imagen_baja,
        'mascara':              mascara,
        'b':                    b.astype(np.float64),
        'forma':                (nrows, ncols),
        'omega_opt':            estimar_omega_optimo(mascara),
        'omega_malla_completa': calcular_omega_optimo(nrows, ncols),
        'factor':               factor,
    }


# Problema 2: gradiente "screened" del modelo RZI

def construir_problema_gradiente(t: np.ndarray, lam: float = 0.3,
                                  beta: float = 1.6, tau: float = 0.05,
                                  sigma: float = 0.6,
                                  campo_desde: np.ndarray = None) -> dict:
    """
    Construye un paso del modelo RZI en la malla fina:

        (4 + lam) u - (suma de vecinos) = -div(v) + lam * t

    t           : estimacion actual en alta resolucion (ya corregida con
                  back-projection). Es el termino de fidelidad y el punto
                  de partida del solver.
    campo_desde : imagen de la que se toma el campo de gradientes v (por
                  defecto, t misma).
    lam         : peso de fidelidad. Pequeno = manda el campo de gradientes
                  (bordes mas firmes); grande = manda t.
    beta, tau, sigma : ver amplified_guidance().

    Retorna el dict que espera core.sor_solver.resolver_screened().
    """
    if t.ndim != 2:
        raise ValueError("Se espera una imagen 2D (un canal).")
    nrows, ncols = t.shape
    fuente = t if campo_desde is None else campo_desde

    gx, gy = amplified_guidance(fuente, beta=beta, tau=tau, sigma_noise=sigma)
    b = -divergence(gx, gy) + lam * t

    return {
        'imagen_baja': t.astype(np.float64),
        'lam_map':     np.full((nrows, ncols), float(lam)),
        'b':           b.astype(np.float64),
        'forma':       (nrows, ncols),
        'omega_opt':   omega_opt_rectangular(nrows, ncols, lam),
        'lam':         float(lam),
    }
