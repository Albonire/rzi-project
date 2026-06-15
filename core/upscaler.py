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

from core.image_ops import resize_float, unsharp_mask_float, ycbcr_a_rgb
from core.edge_enhance import shock_filter, anisotropic_diffusion
from core.poisson_builder import construir_problema_screened
from core.sor_solver import resolver_screened, resolver_screened_con_snapshots


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


# alias retro-compatible: la implementacion vive en core/image_ops (modulo
# compartido sin dependencias circulares con poisson_builder).
_resize_float = resize_float


def back_projection(u: np.ndarray, f_lr: np.ndarray, n_pass: int = 3,
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

    Cada pasada minimiza el residuo ||Down(u) - f_lr||, recuperando alta
    frecuencia de forma principiada ("back-projection iterativa reforzada").
    n_pass=3 (por defecto) converge mejor que 2 sin llegar a oscilar;
    gain<1 amortigua cualquier sobre-correccion entre pasadas.
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


def finalizar_display(u: np.ndarray, tile: np.ndarray, factor: int,
                       usm_pct: float, radius: float = 1.5,
                       threshold: float = 0.0,
                       usar_difusion: bool = True, difusion_iters: int = 5,
                       difusion_kappa: float = 0.03,
                       usar_shock: bool = True, shock_iters: int = 3) -> np.ndarray:
    """
    Acabado de display compartido: lleva 'u' (salida de un solver, en
    cualquier grilla intermedia) al tamano de display tw*factor x th*factor
    con LANCZOS y aplica, en orden, el "stack de nitidez" puramente
    matematico (seccion 5.3 de steps.md):

      1. resize LANCZOS al tamano de display.
      2. anisotropic_diffusion (Perona-Malik), si usar_difusion: limpia
         ruido de grano en zonas planas preservando bordes. Va PRIMERO para
         no amplificar ese ruido en el paso siguiente.
      3. shock_filter (Osher-Rudin), si usar_shock: convierte rampas
         borrosas en transiciones nitidas (afila bordes).
      4. USM (si usm_pct>0): realce adicional opcional, tipicamente bajo
         porque el shock filter ya aporta la mayor parte de la nitidez.
      5. anti_ringing contra una guia bicubica del mismo tamano: acota
         cualquier overshoot introducido por shock/USM (evita halos).

    Todo el calculo se hace en float64 [0,1] (resize_float + unsharp_mask_float
    + edge_enhance de core/, sin viaje intermedio por uint8/256 niveles, lo
    que evita banding al amplificar degradados suaves).

    Es la funcion que usan tanto "Reconstruir" (_upscale_display) como cada
    frame de "Animar convergencia": cualquier mejora aqui se refleja en
    ambos flujos.
    """
    th, tw = tile.shape
    out_h, out_w = th * factor, tw * factor

    u = np.clip(u, 0.0, 1.0)
    if u.shape != (out_h, out_w):
        display = resize_float(u, out_w, out_h, Image.LANCZOS)
    else:
        display = u.copy()

    if usar_difusion and difusion_iters > 0:
        display = anisotropic_diffusion(display, iteraciones=difusion_iters,
                                          kappa=difusion_kappa)

    if usar_shock and shock_iters > 0:
        display = shock_filter(display, iteraciones=shock_iters)

    if usm_pct > 0:
        display = unsharp_mask_float(display, percent=usm_pct, radius=radius,
                                       threshold=threshold)

    bic = resize_float(tile, out_w, out_h, Image.BICUBIC)
    bic = np.clip(bic, 0.0, 1.0)

    return anti_ringing(display, bic)


def finalizar_display_color(y_recon: np.ndarray, cb: np.ndarray, cr: np.ndarray,
                             y_tile: np.ndarray, factor: int, usm_pct: float,
                             radius: float = 1.5, threshold: float = 0.0,
                             usar_difusion: bool = True, difusion_iters: int = 5,
                             difusion_kappa: float = 0.03,
                             usar_shock: bool = True, shock_iters: int = 3) -> np.ndarray:
    """
    Acabado de display para imagenes a color (enfoque luminancia-PDE,
    seccion 5.3 de steps.md).

    El canal de luminancia ('y_recon', salida del solver Poisson-screened)
    pasa por TODO el stack de `finalizar_display` (difusion, shock, USM,
    anti-ringing): la nitidez y la limpieza de ruido se calculan sobre
    luminancia, donde el ojo es mas sensible. Los canales de crominancia
    (cb, cr) solo se reescalan a bicubica al tamano final, sin realce (esto
    evita franjas/artefactos de color que aparecerian si shock/difusion se
    aplicaran por canal RGB de forma independiente).

    Devuelve una imagen RGB float64 [0,1] de forma (th*factor, tw*factor, 3).
    """
    y_disp = finalizar_display(y_recon, y_tile, factor, usm_pct,
                                radius=radius, threshold=threshold,
                                usar_difusion=usar_difusion,
                                difusion_iters=difusion_iters,
                                difusion_kappa=difusion_kappa,
                                usar_shock=usar_shock,
                                shock_iters=shock_iters)

    out_h, out_w = y_disp.shape
    cb_disp = resize_float(cb, out_w, out_h, Image.BICUBIC)
    cr_disp = resize_float(cr, out_w, out_h, Image.BICUBIC)
    cb_disp = np.clip(cb_disp, 0.0, 1.0)
    cr_disp = np.clip(cr_disp, 0.0, 1.0)

    return ycbcr_a_rgb(y_disp, cb_disp, cr_disp)


def upscale_poisson_screened(tile: np.ndarray, factor: int = 4,
                              lam: float = 8.0, beta: float = 1.7,
                              tau: float = 0.02,
                              usar_back_projection: bool = True,
                              usar_anti_ringing: bool = True,
                              usar_multiescala: bool = True,
                              n_back_pass: int = 3, back_gain: float = 0.8,
                              tol: float = 1e-4, max_iter: int = 600,
                              capturar_snapshots: bool = False,
                              snapshots_cada: int = 8,
                              max_snapshots: int = 48) -> dict:
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

    Si capturar_snapshots=True, se usa resolver_screened_con_snapshots en cada
    etapa de la cascada y se acumulan los estados intermedios de 'u' (cada
    etapa en su propia grilla) en 'snapshots', agregando 'imagen_recon' final
    (tras back-projection y anti-ringing) como ultimo elemento. Si la lista
    resultante supera 'max_snapshots', se submuestrea uniformemente
    conservando el primero y el ultimo. Sin capturar_snapshots, el resultado
    es identico al de antes ('snapshots' es None).

    Retorna un dict:
        'imagen_recon' : ndarray float64 [0,1], imagen agrandada.
        'iteraciones'  : suma de iteraciones SOR de todas las etapas.
        'tiempo'       : tiempo total de SOR (sin contar resize/filtros).
        'omega'        : omega usado en la ultima etapa.
        'convergio'    : True si todas las etapas convergieron.
        'etapas'       : lista de factores de cada etapa de la cascada.
        'lam'          : lambda usado.
        'snapshots'    : lista de estados intermedios de 'u' (o None).
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
    snapshots_raw = [] if capturar_snapshots else None

    for etapa_factor in etapas:
        if etapa_factor <= 1:
            continue
        problema = construir_problema_screened(u, etapa_factor, lam=lam,
                                                 beta=beta, tau=tau)
        if capturar_snapshots:
            resultado = resolver_screened_con_snapshots(
                problema, tol=tol, max_iter=max_iter, cada_n_iter=snapshots_cada)
            snapshots_raw.extend(resultado['snapshots'])
        else:
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

    snapshots = None
    if capturar_snapshots:
        snapshots_raw.append(np.clip(u, 0.0, 1.0))
        if len(snapshots_raw) > max_snapshots:
            idx = np.linspace(0, len(snapshots_raw) - 1, max_snapshots).round().astype(int)
            idx = sorted(set(idx.tolist()))
            snapshots_raw = [snapshots_raw[i] for i in idx]
        snapshots = snapshots_raw

    return {
        'imagen_recon': np.clip(u, 0.0, 1.0),
        'iteraciones':  iters_total,
        'tiempo':       tiempo_total,
        'omega':        omega_usado,
        'convergio':    convergio_total,
        'historial':    historial_final,
        'etapas':       etapas,
        'lam':          lam,
        'snapshots':    snapshots,
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
