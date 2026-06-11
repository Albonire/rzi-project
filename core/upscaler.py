"""
Capa 2 (extension) - Pipeline de upscaling con Poisson "screened".

Implementa la seccion C del documento "Analisis Critico y Mejora del
Proyecto RZI": cascada multiescala (coarse-to-fine), back-projection y
anti-ringing, ademas de una variante para imagenes a color (YCbCr).

Usa PIL (modo 'F' para resize de arreglos float) y scipy.ndimage; no agrega
cv2 como dependencia nueva del proyecto.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter

from core.poisson_builder import construir_problema_screened
from core.sor_solver import resolver_screened


def _descomponer_factor(factor: int) -> list:
    """
    Descompone 'factor' en una secuencia de etapas para la cascada
    coarse-to-fine.

    Para factores potencia de 2 (2, 4, 8, ...) produce una cascada completa
    de etapas x2. Para factor=6 produce [2, 3] (una etapa x2 seguida de una
    x3). Para factores sin divisores pares utiles (3, 5, 7, ...) o factor=1
    devuelve una unica etapa [factor] (resolucion directa, sin cascada).
    """
    if factor <= 1:
        return [max(1, factor)]

    etapas = []
    f = factor
    while f % 2 == 0 and f > 2:
        etapas.append(2)
        f //= 2
    etapas.append(f)
    return etapas


def _resize_float(arr: np.ndarray, new_w: int, new_h: int, metodo) -> np.ndarray:
    """Redimensiona un arreglo float64 [0,1] preservando precision (modo 'F' de PIL)."""
    pil = Image.fromarray(arr.astype(np.float32), mode='F')
    pil = pil.resize((new_w, new_h), metodo)
    return np.asarray(pil, dtype=np.float64)


def back_projection(u: np.ndarray, f_lr: np.ndarray, n_pass: int = 2,
                     gain: float = 0.8) -> np.ndarray:
    """
    Refuerza la consistencia Down(u) ~= f_lr (seccion C del documento).

    En cada pasada:
      1. Down(u): se reduce 'u' al tamano de 'f_lr' con Image.BOX (promedio
         de area, evita aliasing).
      2. residuo = f_lr - Down(u).
      3. Up(residuo): se agranda el residuo de vuelta al tamano de 'u' con
         Image.BICUBIC.
      4. u <- u + gain * Up(residuo).

    n_pass=2-3 pasadas suelen bastar; gain<1 evita oscilaciones.
    """
    h_lr, w_lr = f_lr.shape
    h_hi, w_hi = u.shape

    out = u.copy()
    for _ in range(n_pass):
        bajada  = _resize_float(out, w_lr, h_lr, Image.BOX)
        residuo = f_lr - bajada
        subida  = _resize_float(residuo, w_hi, h_hi, Image.BICUBIC)
        out = out + gain * subida

    return out


def anti_ringing(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Recorta 'u' al rango [min, max] local 3x3 de una referencia bicubica
    'ref', evitando el sobre-realce ("ringing") cerca de bordes fuertes
    (seccion C del documento).
    """
    lo = minimum_filter(ref, size=3)
    hi = maximum_filter(ref, size=3)
    return np.clip(u, lo, hi)


def upscale_poisson_screened(tile: np.ndarray, factor: int = 4,
                              lam: float = 8.0, beta: float = 1.7,
                              tau: float = 0.02,
                              usar_back_projection: bool = True,
                              usar_anti_ringing: bool = True,
                              usar_multiescala: bool = True,
                              n_back_pass: int = 2, back_gain: float = 0.8,
                              tol: float = 1e-4, max_iter: int = 600) -> dict:
    """
    Pipeline completo de upscaling con Poisson "screened" + amplificacion
    adaptativa de gradientes (seccion B y C del documento).

    Si usar_multiescala=True y 'factor' admite descomposicion en etapas mas
    pequenas (ver _descomponer_factor), se resuelve en cascada
    coarse-to-fine: el resultado de cada etapa sirve de entrada (warm start)
    a la siguiente. Esto mejora la calidad y reduce el numero total de
    iteraciones de SOR respecto de resolver directamente al factor final.

    Tras la cascada se aplican (si estan activados):
      - back_projection: fuerza Down(u) ~= tile original.
      - anti_ringing: recorta a los limites locales de la guia bicubica
        final, evitando sobre-realces.

    Retorna un dict:
        'imagen_recon' : ndarray float64 [0,1], imagen agrandada.
        'iteraciones'  : suma de iteraciones SOR de todas las etapas.
        'tiempo'       : tiempo total de SOR (sin contar resize/filtros).
        'omega'        : omega usado en la ultima etapa.
        'convergio'    : True si todas las etapas convergieron.
        'etapas'       : lista de factores de cada etapa de la cascada.
        'lam'          : lambda usado.
    """
    if tile.ndim != 2:
        raise ValueError("Se espera tile 2D en escala de grises.")

    f_lr = tile.astype(np.float64)

    if usar_multiescala:
        etapas = _descomponer_factor(factor)
    else:
        etapas = [factor]

    u = f_lr
    iters_total = 0
    tiempo_total = 0.0
    omega_usado = None
    historial_final = np.array([])
    convergio_total = True

    for etapa_factor in etapas:
        if etapa_factor <= 1:
            continue
        problema = construir_problema_screened(u, etapa_factor, lam=lam,
                                                 beta=beta, tau=tau)
        resultado = resolver_screened(problema, tol=tol, max_iter=max_iter)
        u = resultado['imagen_recon']
        iters_total += resultado['iteraciones']
        tiempo_total += resultado['tiempo']
        omega_usado = resultado['omega']
        historial_final = resultado['historial']
        convergio_total = convergio_total and resultado['convergio']

    if usar_back_projection:
        u = back_projection(u, f_lr, n_pass=n_back_pass, gain=back_gain)

    if usar_anti_ringing:
        # La malla real de u puede ser menor que la nominal (h-1)*factor+1 si el
        # tope de memoria de construir_problema_screened redujo el tile durante la
        # cascada. Construimos la guia al tamaño REAL de u para que las formas
        # coincidan (evita el crash de broadcast en regiones grandes).
        new_h, new_w = u.shape
        ref = _resize_float(f_lr, new_w, new_h, Image.BICUBIC)
        u = anti_ringing(u, ref)

    return {
        'imagen_recon': np.clip(u, 0.0, 1.0),
        'iteraciones':  iters_total,
        'tiempo':       tiempo_total,
        'omega':        omega_usado,
        'convergio':    convergio_total,
        'historial':    historial_final,
        'etapas':       etapas,
        'lam':          lam,
    }


def upscale_color(rgb_arr: np.ndarray, factor: int = 4, lam: float = 8.0,
                   beta: float = 1.7, tau: float = 0.02,
                   **kwargs) -> np.ndarray:
    """
    Upscaling de imagenes a color (seccion C del documento): el modelo
    Poisson "screened" se resuelve solo sobre el canal de luminancia (Y de
    YCbCr), mientras que los canales de crominancia (Cb, Cr) se agrandan con
    interpolacion bicubica (el ojo es mucho menos sensible a la nitidez del
    color que a la de la luminancia).

    Esta funcion es una capacidad de la libreria que NO esta conectada a la
    interfaz grafica (que trabaja solo en escala de grises); se documenta
    aqui para uso programatico o futuras extensiones de la UI.

    Parametros
    ----------
    rgb_arr : ndarray float64 [0,1] o uint8, shape (h, w, 3).
    factor, lam, beta, tau, **kwargs : ver upscale_poisson_screened.

    Retorna
    -------
    ndarray float64 [0,1], shape (new_h, new_w, 3).
    """
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
        raise ValueError("Se espera imagen RGB de 3 canales.")

    img = np.asarray(rgb_arr, dtype=np.float64)
    if img.max() > 1.01:
        img = img / 255.0

    pil_rgb = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8),
                               mode='RGB')
    y, cb, cr = pil_rgb.convert('YCbCr').split()

    y_arr = np.asarray(y, dtype=np.float64) / 255.0

    resultado = upscale_poisson_screened(y_arr, factor=factor, lam=lam,
                                          beta=beta, tau=tau, **kwargs)
    y_up = resultado['imagen_recon']
    new_h, new_w = y_up.shape

    cb_up = cb.resize((new_w, new_h), Image.BICUBIC)
    cr_up = cr.resize((new_w, new_h), Image.BICUBIC)
    y_up_pil = Image.fromarray((np.clip(y_up, 0.0, 1.0) * 255).astype(np.uint8),
                                mode='L')

    salida = Image.merge('YCbCr', (y_up_pil, cb_up, cr_up)).convert('RGB')
    return np.asarray(salida, dtype=np.float64) / 255.0
