"""
Capa 2 (extension) - Operaciones de imagen en float64 [0,1].

Modulo de responsabilidad unica: utilidades de reescalado y realce que
trabajan SIEMPRE en float64, sin pasar por una cuantizacion intermedia a
uint8 (256 niveles). Lo usan tanto `core/poisson_builder.py` (guia bicubica
del solver) como `core/upscaler.py` (acabado de display), evitando el
banding/posterizacion que aparece cuando un degradado suave se cuantiza a 8
bits antes de amplificarlo o realzarlo.

No depende de poisson_builder ni de upscaler (evita imports circulares):
ambos importan de aqui.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def resize_float(arr: np.ndarray, new_w: int, new_h: int, metodo) -> np.ndarray:
    """
    Redimensiona un arreglo float64 [0,1] preservando precision.

    Usa el modo 'F' (float32) de PIL para el resize: a diferencia de
    convertir a uint8 antes de escalar, no introduce escalones de
    cuantizacion (256 niveles) en degradados suaves.
    """
    pil = Image.fromarray(arr.astype(np.float32), mode='F')
    pil = pil.resize((new_w, new_h), metodo)
    return np.asarray(pil, dtype=np.float64)


def unsharp_mask_float(img: np.ndarray, percent: float, radius: float = 1.5,
                        threshold: float = 0.0) -> np.ndarray:
    """
    Unsharp mask en float64 [0,1].

    Equivalente a PIL.ImageFilter.UnsharpMask pero sin cuantizar a 8 bits:
        blur = gaussian_filter(img, sigma=radius)
        mask = img - blur
        mask = 0 donde |mask| < threshold   (umbral en unidades [0,1])
        out  = clip(img + (percent/100) * mask, 0, 1)

    'percent' usa la misma convencion que UnsharpMask de PIL (100 = ganancia
    unitaria de la mascara). 'threshold' esta en unidades [0,1] (PIL usa
    [0,255]; threshold=2/255 ~ 0.008 seria el equivalente a threshold=2).
    """
    if percent <= 0:
        return img

    blur = gaussian_filter(img, sigma=radius)
    mask = img - blur

    if threshold > 0:
        mask = np.where(np.abs(mask) < threshold, 0.0, mask)

    out = img + (percent / 100.0) * mask
    return np.clip(out, 0.0, 1.0)


# Matriz BT.601 RGB -> YCbCr (con Cb, Cr centrados en 0.5), todo en [0,1]
_RGB_A_YCBCR = np.array([
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312],
])
_YCBCR_A_RGB = np.linalg.inv(_RGB_A_YCBCR)


def rgb_a_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte una imagen RGB float64 [0,1] (H,W,3) a sus tres canales
    Y, Cb, Cr (cada uno (H,W), tambien en float64 [0,1]).

    Usa la matriz BT.601 directamente en float, sin pasar por uint8: evita
    el banding que introduciria una cuantizacion intermedia a 8 bits, tanto
    en la luminancia como en el croma.

    Cb y Cr quedan centrados en 0.5 (en vez de 0), siguiendo la convencion
    habitual de codecs de imagen, para poder tratarlos como canales [0,1]
    igual que Y.
    """
    ycbcr = rgb @ _RGB_A_YCBCR.T
    y = ycbcr[..., 0]
    cb = ycbcr[..., 1] + 0.5
    cr = ycbcr[..., 2] + 0.5
    return y, cb, cr


def ycbcr_a_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """
    Inversa de `rgb_a_ycbcr`: combina los canales Y, Cb, Cr (cada uno
    (H,W) en float64 [0,1]) en una imagen RGB float64 (H,W,3), clipeada a
    [0,1].
    """
    ycbcr = np.stack([y, cb - 0.5, cr - 0.5], axis=-1)
    rgb = ycbcr @ _YCBCR_A_RGB.T
    return np.clip(rgb, 0.0, 1.0)


def resize_rgb_float(rgb: np.ndarray, new_w: int, new_h: int, metodo) -> np.ndarray:
    """
    Reescala una imagen RGB float64 [0,1] (H,W,3) a (new_h, new_w, 3),
    aplicando `resize_float` canal por canal (mismo metodo PIL para los
    tres canales, p. ej. Image.BICUBIC o Image.NEAREST).
    """
    canales = [resize_float(rgb[..., c], new_w, new_h, metodo) for c in range(rgb.shape[-1])]
    return np.clip(np.stack(canales, axis=-1), 0.0, 1.0)
