"""
Capa 2 (extension) - Operadores de realce perceptual en float64 [0,1].

Modulo de responsabilidad unica: operadores PDE explicitos (sin redes
neuronales, sin datos aprendidos) que se usan en el acabado de display
(`core/upscaler.py::finalizar_display`) para afilar bordes y limpiar ruido
despues de la reconstruccion Poisson-screened.

No depende de poisson_builder ni de upscaler (evita imports circulares):
ambos pueden importar de aqui.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def shock_filter(u: np.ndarray, iteraciones: int = 3, dt: float = 0.25,
                  sigma: float = 1.0) -> np.ndarray:
    """
    Filtro de shock de Osher-Rudin / Alvarez-Mazorra.

    Evoluciona la imagen segun:
        u_t = -sign(G_sigma * Laplaciano(u)) * |grad u|

    donde G_sigma * Laplaciano(u) es el laplaciano suavizado con un
    gaussiano de desviacion `sigma` (evita disparar shocks por ruido de
    alta frecuencia). El esquema es Euler explicito con `iteraciones` pasos
    de tamano `dt`.

    Efecto: convierte rampas (transiciones borrosas) en escalones nitidos,
    "empujando" cada punto hacia el lado de la rampa que le corresponde
    segun la curvatura local. En zonas planas |grad u| ~ 0, por lo que casi
    no se modifican (no introduce escalones artificiales en degradados
    suaves). Es un operador puramente local y determinista, sin
    entrenamiento ni datos externos.

    Devuelve float64 en [0,1] con la misma forma que `u`.
    """
    if iteraciones <= 0:
        return np.clip(u, 0.0, 1.0)

    out = u.astype(np.float64).copy()

    for _ in range(iteraciones):
        # Gradiente centrado (bordes con diferencias hacia adentro)
        gy, gx = np.gradient(out)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Laplaciano (suma de segundas derivadas) suavizado con gaussiano
        lap = (
            np.roll(out, 1, axis=0) + np.roll(out, -1, axis=0) +
            np.roll(out, 1, axis=1) + np.roll(out, -1, axis=1) - 4.0 * out
        )
        lap_suave = gaussian_filter(lap, sigma=sigma)

        out = out - dt * np.sign(lap_suave) * grad_mag

    return np.clip(out, 0.0, 1.0)


def anisotropic_diffusion(u: np.ndarray, iteraciones: int = 5,
                           kappa: float = 0.03, dt: float = 0.2) -> np.ndarray:
    """
    Difusion anisotropica de Perona-Malik (esquema explicito, 4 vecinos).

    Para cada vecino N/S/E/O calcula la diferencia local `d` y una funcion
    de parada de borde:
        g(d) = 1 / (1 + (d / kappa)^2)

    que vale ~1 donde el gradiente es pequeno (zonas planas, difunde y por
    tanto suaviza ruido) y ~0 donde el gradiente es grande (bordes, no
    difunde y por tanto los preserva). La actualizacion explicita es:
        u <- u + dt * sum_dir g(d_dir) * d_dir

    Efecto: limpia ruido tipo "grano" en regiones homogeneas sin difuminar
    los contornos. `kappa` controla que tan grande debe ser un gradiente
    para considerarse "borde" (valores tipicos 0.02-0.05 en escala [0,1]).

    Devuelve float64 en [0,1] con la misma forma que `u`.
    """
    if iteraciones <= 0:
        return np.clip(u, 0.0, 1.0)

    out = u.astype(np.float64).copy()

    for _ in range(iteraciones):
        d_n = np.roll(out, 1, axis=0) - out
        d_s = np.roll(out, -1, axis=0) - out
        d_e = np.roll(out, -1, axis=1) - out
        d_w = np.roll(out, 1, axis=1) - out

        g_n = 1.0 / (1.0 + (d_n / kappa) ** 2)
        g_s = 1.0 / (1.0 + (d_s / kappa) ** 2)
        g_e = 1.0 / (1.0 + (d_e / kappa) ** 2)
        g_w = 1.0 / (1.0 + (d_w / kappa) ** 2)

        out = out + dt * (g_n * d_n + g_s * d_s + g_e * d_e + g_w * d_w)

    return np.clip(out, 0.0, 1.0)
