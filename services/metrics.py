"""
Capa 3 - Metricas de calidad, protocolo de degradacion y lineas base.

Con verdad de campo (modo Comparar):
  PSNR  : error pixel a pixel, en dB. Mas alto es mejor.
  SSIM  : parecido de estructura local (brillo, contraste, forma). 1 = igual.
  GMSD  : diferencia de bordes (magnitud de gradiente). Mas BAJO es mejor.

Sin verdad de campo (modo Reconstruir):
  nitidez_gradiente  : magnitud media del gradiente (mas alto = mas nitido).
  consistencia_datos : PSNR entre Down(u) y la imagen pequena de entrada
                       (alto = la ampliacion respeta los datos originales).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve

from core.image_ops import resize_float

try:
    from skimage.metrics import (peak_signal_noise_ratio as _psnr_fn,
                                 structural_similarity as _ssim_fn)
    SKIMAGE_DISPONIBLE = True
except ImportError:
    SKIMAGE_DISPONIBLE = False


# Metricas con referencia

def psnr(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """PSNR = 10 * log10(1 / MSE) para imagenes en [0,1]."""
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstruida, dtype=np.float64)
    mse = np.mean((o - r) ** 2)
    if mse < 1e-15:
        return float('inf')
    if SKIMAGE_DISPONIBLE:
        return float(_psnr_fn(o, r, data_range=1.0))
    return float(10.0 * np.log10(1.0 / mse))


def ssim(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """SSIM (Wang et al., 2004). Usa scikit-image si esta disponible."""
    o = np.asarray(original, dtype=np.float64)
    r = np.asarray(reconstruida, dtype=np.float64)
    if SKIMAGE_DISPONIBLE and min(o.shape[:2]) >= 7:
        return float(_ssim_fn(o, r, data_range=1.0))
    # Version global simplificada (respaldo)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_o, mu_r = np.mean(o), np.mean(r)
    var_o, var_r = np.var(o), np.var(r)
    cov = np.mean((o - mu_o) * (r - mu_r))
    num = (2 * mu_o * mu_r + c1) * (2 * cov + c2)
    den = (mu_o ** 2 + mu_r ** 2 + c1) * (var_o + var_r + c2)
    return float(num / den)


def _magnitud_prewitt(img: np.ndarray) -> np.ndarray:
    px = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float64) / 3.0
    gx = convolve(img, px, mode='nearest')
    gy = convolve(img, px.T, mode='nearest')
    return np.sqrt(gx ** 2 + gy ** 2)


def gmsd(original: np.ndarray, reconstruida: np.ndarray) -> float:
    """
    Gradient Magnitude Similarity Deviation (Xue et al., 2014).
    Compara los mapas de bordes de ambas imagenes; 0 = identicas.
    """
    m_o = _magnitud_prewitt(np.asarray(original, dtype=np.float64))
    m_r = _magnitud_prewitt(np.asarray(reconstruida, dtype=np.float64))
    c = 0.0026
    gms = (2 * m_o * m_r + c) / (m_o ** 2 + m_r ** 2 + c)
    return float(np.std(gms))


# Metricas sin referencia

def nitidez_gradiente(img: np.ndarray) -> float:
    """Magnitud media del gradiente (diferencias centradas)."""
    gy, gx = np.gradient(np.asarray(img, dtype=np.float64))
    return float(np.mean(np.hypot(gx, gy)))


def consistencia_datos(ampliada: np.ndarray, pequena: np.ndarray) -> float:
    """PSNR entre Down(ampliada) y la imagen pequena (en dB)."""
    h, w = pequena.shape
    return psnr(pequena, np.clip(resize_float(ampliada, w, h, Image.BICUBIC), 0, 1))


# Protocolo de degradacion y lineas base

def degradar(hr: np.ndarray, factor: int) -> tuple:
    """
    Protocolo unico de prueba con verdad de campo:
      1. recorta la imagen para que sus lados sean multiplos de 'factor';
      2. la reduce 'factor' veces con bicubica + antialias en float64 (el
         protocolo estandar en la literatura de super-resolucion).
    Retorna (hr_recortada, lr).
    """
    hr = np.asarray(hr, dtype=np.float64)
    H, W = hr.shape[:2]
    H, W = H - H % factor, W - W % factor
    hr = hr[:H, :W]
    lr = np.clip(resize_float(hr, W // factor, H // factor, Image.BICUBIC), 0.0, 1.0)
    return hr, lr


def interpolacion_bicubica(lr: np.ndarray, factor: int) -> np.ndarray:
    """Linea base de la industria: ampliacion bicubica a tamano exacto."""
    h, w = lr.shape
    return np.clip(resize_float(lr, w * factor, h * factor, Image.BICUBIC), 0.0, 1.0)


def interpolacion_lanczos(lr: np.ndarray, factor: int) -> np.ndarray:
    """Segunda linea base: Lanczos (ventana de 3 lobulos), tamano exacto."""
    h, w = lr.shape
    return np.clip(resize_float(lr, w * factor, h * factor, Image.LANCZOS), 0.0, 1.0)


# Tabla comparativa

def tabla_comparativa(original: np.ndarray, candidatas: dict,
                      modelos_rzi: tuple = ()) -> list:
    """
    Calcula PSNR, SSIM, GMSD y nitidez de cada imagen candidata contra la
    verdad de campo.

    candidatas  : dict {nombre: imagen} (en el orden en que se mostraran).
    modelos_rzi : nombres que corresponden al modelo RZI (para el CSV).
    """
    filas = []
    for nombre, img in candidatas.items():
        filas.append({
            'modelo':  nombre,
            'psnr':    psnr(original, img),
            'ssim':    ssim(original, img),
            'gmsd':    gmsd(original, img),
            'nitidez': nitidez_gradiente(img),
            'es_rzi':  nombre in modelos_rzi,
        })
    return filas
