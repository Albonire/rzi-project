"""
Capa 3 - Metricas de Calidad y Comparativas.

Implementa:
  - PSNR y SSIM para medir calidad de reconstruccion (RF05, seccion 8.2).
  - Interpolacion bicubica como linea base de la industria (RF04).
  - Tabla comparativa: Original / Bicubica / Laplace / Poisson SOR.
"""

import numpy as np
from PIL import Image

try:
    from skimage.metrics import (peak_signal_noise_ratio as _psnr_fn,
                                  structural_similarity as _ssim_fn)
    SKIMAGE_DISPONIBLE = True
except ImportError:
    SKIMAGE_DISPONIBLE = False


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def psnr(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """
    Peak Signal-to-Noise Ratio en decibeles.

    PSNR = 10 * log10(MAX^2 / MSE)
    Para imagenes normalizadas a [0,1], MAX = 1.0.
    Valores mas altos indican mejor calidad. >30 dB es bueno, >40 dB es excelente.
    """
    o = np.asarray(original,     dtype=np.float64)
    r = np.asarray(reconstruida, dtype=np.float64)
    if SKIMAGE_DISPONIBLE:
        return float(_psnr_fn(o, r, data_range=1.0))
    mse = np.mean((o - r) ** 2)
    if mse < 1e-15:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def ssim(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """
    Structural Similarity Index en [-1, 1].

    Mide similitud perceptual (brillo, contraste y estructura).
    1.0 = identicas, 0.0 = sin correlacion, <0 = inversas.
    Un SSIM alto con bajo PSNR indica que se preserva la estructura
    pero hay diferencias de nivel de gris (lo que ocurre con Laplace).
    """
    o = np.asarray(original,     dtype=np.float64)
    r = np.asarray(reconstruida, dtype=np.float64)
    if SKIMAGE_DISPONIBLE:
        return float(_ssim_fn(o, r, data_range=1.0))
    # Implementacion simplificada de fallback
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_o, mu_r = np.mean(o), np.mean(r)
    var_o, var_r = np.var(o), np.var(r)
    cov = np.mean((o - mu_o) * (r - mu_r))
    num = (2 * mu_o * mu_r + c1) * (2 * cov + c2)
    den = (mu_o**2 + mu_r**2 + c1) * (var_o + var_r + c2)
    return float(num / den)


# ---------------------------------------------------------------------------
# Linea base: interpolacion bicubica (RF04)
# ---------------------------------------------------------------------------

def interpolacion_bicubica(imagen_original: np.ndarray,
                            factor: int) -> np.ndarray:
    """
    Genera la reconstruccion bicubica de referencia (RF04).

    Proceso:
      1. Toma la imagen original float64 [0,1].
      2. La reduce al tamano 1/factor con LANCZOS (anti-aliasing).
      3. La expande de vuelta al tamano original con interpolacion BICUBIC.

    Este es el metodo estandar de la industria para comparar contra el
    modelo matematico Poisson SOR (tabla de la seccion 8.2).
    """
    orig = np.clip(imagen_original, 0.0, 1.0)
    nrows, ncols = orig.shape

    img_uint8 = (orig * 255).astype(np.uint8)
    pil_orig  = Image.fromarray(img_uint8, mode='L')

    # Reducir
    w_baja = max(2, ncols // factor)
    h_baja = max(2, nrows // factor)
    pil_baja = pil_orig.resize((w_baja, h_baja), Image.LANCZOS)

    # Expandir con bicubico
    pil_bicubic = pil_baja.resize((ncols, nrows), Image.BICUBIC)

    return np.asarray(pil_bicubic, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Tabla comparativa (seccion 8.2)
# ---------------------------------------------------------------------------

def tabla_comparativa(original:   np.ndarray,
                      bicubica:   np.ndarray,
                      laplace:    np.ndarray,
                      poisson_sor: np.ndarray) -> list:
    """
    Calcula PSNR y SSIM para los tres metodos de reconstruccion.

    Retorna una lista de dicts lista para mostrar en la interfaz o exportar a CSV.
    """
    return [
        {
            'modelo': 'Bicubica (Pillow)',
            'psnr':   psnr(original, bicubica),
            'ssim':   ssim(original, bicubica),
        },
        {
            'modelo': 'Laplace Puro (b=0)',
            'psnr':   psnr(original, laplace),
            'ssim':   ssim(original, laplace),
        },
        {
            'modelo': 'Poisson SOR Red-Black',
            'psnr':   psnr(original, poisson_sor),
            'ssim':   ssim(original, poisson_sor),
        },
    ]
