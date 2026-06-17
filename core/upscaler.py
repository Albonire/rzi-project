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


def back_projection(u: np.ndarray, f_lr: np.ndarray, n_pass: int = 2,
                     gain: float = 0.6) -> np.ndarray:
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
    frecuencia de forma principiada ("back-projection iterativa").
    n_pass=2 y gain=0.6 (por defecto) equilibran fidelidad y ruido: el paso
    re-inyecta las altas frecuencias del tile original, incluyendo grano; un
    gain menor reduce ese ruido sin sacrificar la mejora de fidelidad.
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


def generar_frames_pipeline(
    canal: np.ndarray, factor: int,
    *,
    lam: float = 8.0, beta: float = 1.7, tau: float = 0.02,
    usar_back_projection: bool = True,
    usar_anti_ringing: bool = True,
    usar_multiescala: bool = True,
    usar_difusion: bool = True,
    usar_shock: bool = True,
    usm_pct: float = 60.0,
    tol: float = 1e-4, max_iter: int = 600,
    snapshots_cada: int = 8,
    max_snaps_por_etapa: int = 14,
    n_back_pass: int = 2, back_gain: float = 0.6,
) -> dict:
    """
    Genera la secuencia didactica completa de frames para "Animar Convergencia".

    Recorre TODO el pipeline RZI en orden, capturando un frame en cada fase:
    imagen original (NN), interpolacion bicubica, cascada SOR por etapa,
    back-projection, anti-ringing inicial, difusion anisotropica, filtro de
    shock, USM opcional, y reconstruccion final.

    El ultimo frame es IDENTICO al que produce finalizar_display() con los
    mismos parametros, usando las mismas primitivas en el mismo orden.

    Todos los frames tienen forma (th*factor, tw*factor), float64 [0,1].

    Retorna un dict:
        'frames'         : lista de ndarrays 2D float64 [0,1] a tamano display.
        'pasos'          : lista de nombres de los pasos del proceso (para la
                           lista lateral del dialogo).
        'pasos_de_frame' : indice en 'pasos' de cada frame (para resaltar el
                           paso actual en el listado).
        'subtitulos'     : leyenda descriptiva de cada frame.
        'omega'          : omega SOR de la ultima etapa.
        'iteraciones'    : total de iteraciones SOR.
        'tiempo'         : tiempo total de SOR (sin resize/filtros).
        'etapas'         : lista de factores de cada etapa de la cascada.
    """
    th, tw = canal.shape
    out_h, out_w = th * factor, tw * factor
    f_lr = canal.astype(np.float64)

    frames       = []
    pasos        = []
    pasos_idx    = []   # indice en 'pasos' para cada frame
    subtitulos   = []

    def agregar(frame, paso, subtitulo):
        f_clip = np.clip(frame.astype(np.float64), 0.0, 1.0)
        if paso not in pasos:
            pasos.append(paso)
        pasos_idx.append(pasos.index(paso))
        frames.append(f_clip)
        subtitulos.append(subtitulo)

    # --- Fase 1: interpolacion (creacion de pixeles) ---
    nn = resize_float(f_lr, out_w, out_h, Image.NEAREST)
    agregar(nn, 'Imagen original / vecino mas cercano',
            'Imagen de entrada ampliada con vecino mas cercano: se ven los '
            'pocos pixeles conocidos. NN no crea nueva informacion.')
    agregar(nn, 'Imagen original / vecino mas cercano',
            'NN replica cada pixel LR en un bloque solido: aspecto pixelado, '
            'sin suavizado. Esta es la referencia de partida.')

    bic_display = np.clip(resize_float(f_lr, out_w, out_h, Image.BICUBIC), 0.0, 1.0)
    agregar(bic_display, 'Interpolacion bicubica (guia G)',
            'Bicubica: crea pixeles nuevos usando los 16 vecinos mas proximos '
            'con pesos cubicos. Imagen suave pero sin detalle fino. '
            'Esta guia G es el punto de partida del solver Poisson.')

    # --- Fase 2: cascada SOR screened ---
    if usar_multiescala:
        etapas = _descomponer_factor(factor)
    else:
        etapas = [factor]

    u = f_lr.copy()
    iters_total  = 0
    tiempo_total = 0.0
    omega_usado  = 1.0
    convergio_total = True
    factor_acum  = 1

    for i_et, etapa_factor in enumerate(etapas):
        if etapa_factor <= 1:
            continue

        factor_acum *= etapa_factor
        nombre_paso = f'SOR Screened - etapa {i_et + 1} (escala x{factor_acum})'

        problema = construir_problema_screened(u, etapa_factor,
                                               lam=lam, beta=beta, tau=tau)
        resultado = resolver_screened_con_snapshots(
            problema, tol=tol, max_iter=max_iter,
            cada_n_iter=snapshots_cada)

        snaps_orig  = resultado['snapshots']
        n_orig      = len(snaps_orig)
        iter_nums_o = [k * snapshots_cada for k in range(n_orig)]

        if n_orig > max_snaps_por_etapa:
            sel = np.linspace(0, n_orig - 1, max_snaps_por_etapa).round().astype(int)
            sel = sorted(set(sel.tolist()))
            snaps    = [snaps_orig[i] for i in sel]
            iters_sn = [iter_nums_o[i] for i in sel]
        else:
            snaps    = snaps_orig
            iters_sn = iter_nums_o

        omega_et = resultado['omega']
        for snap, n_it in zip(snaps, iters_sn):
            vis = resize_float(snap, out_w, out_h, Image.LANCZOS)
            agregar(vis, nombre_paso,
                    f'SOR Red-Black convergiendo | iter ~{n_it} | '
                    f'omega = {omega_et:.3f} | '
                    f'Poisson screened: (4+lam)*u = b + lam*G')

        u             = resultado['imagen_recon']
        iters_total  += resultado['iteraciones']
        tiempo_total += resultado['tiempo']
        omega_usado   = omega_et
        convergio_total = convergio_total and resultado['convergio']

    # --- Fase 3: back-projection ---
    if usar_back_projection:
        u_bp = back_projection(u, f_lr, n_pass=n_back_pass, gain=back_gain)
        vis_bp = resize_float(np.clip(u_bp, 0.0, 1.0), out_w, out_h, Image.LANCZOS)
        agregar(vis_bp, 'Back-projection',
                f'Correccion iterativa ({n_back_pass} pasadas, gain={back_gain:.2f}): '
                f'fuerza Down(u) ~= tile original. '
                f'Cada pasada aplica u <- u + gain * Up(tile - Down(u)).')
        u = u_bp

    # --- Fase 4: anti-ringing inicial (escala del solver) ---
    if usar_anti_ringing:
        new_h, new_w = u.shape
        ref_solver = _resize_float(f_lr, new_w, new_h, Image.BICUBIC)
        u_ar = anti_ringing(u, ref_solver)
        vis_ar = resize_float(np.clip(u_ar, 0.0, 1.0), out_w, out_h, Image.LANCZOS)
        agregar(vis_ar, 'Anti-ringing inicial',
                'Recorta u al rango [min, max] local 3x3 de la guia bicubica: '
                'elimina el overshoot cerca de bordes sin borrar el realce.')
        u = u_ar
    elif not usar_back_projection:
        # sin BP ni AR: mostrar la salida cruda del solver a escala display
        vis_raw = resize_float(np.clip(u, 0.0, 1.0), out_w, out_h, Image.LANCZOS)
        agregar(vis_raw, 'Salida del solver SOR',
                'Resultado del solver antes del realce perceptual: '
                'buena fidelidad matematica, todavia algo suave.')

    # --- Preparar imagen a tamano display (LANCZOS, primer paso de finalizar_display) ---
    u = np.clip(u, 0.0, 1.0)
    if u.shape != (out_h, out_w):
        display = resize_float(u, out_w, out_h, Image.LANCZOS)
    else:
        display = u.copy()

    # --- Fase 5: difusion anisotropica (denoise, Perona-Malik) ---
    if usar_difusion:
        display = anisotropic_diffusion(display)
        agregar(display, 'Difusion anisotropica (denoise)',
                'Perona-Malik: g(d) = 1/(1+(d/kappa)^2). '
                'Difunde dentro de regiones homogeneas (g~1) y no en bordes (g~0). '
                'Limpia grano sin difuminar contornos.')

    # --- Fase 6: filtro de shock (Osher-Rudin) ---
    if usar_shock:
        display = shock_filter(display)
        agregar(display, 'Filtro de shock (nitidez de bordes)',
                'Osher-Rudin: u_t = -sign(Lap_sigma(u)) * |grad u|. '
                'Convierte rampas borrosas en transiciones nitidas; '
                'en zonas planas (|grad u|~0) no actua.')

    # --- Fase 7: realce USM (opcional) ---
    if usm_pct > 0:
        display = unsharp_mask_float(display, percent=usm_pct)
        agregar(display, 'Realce USM',
                f'Unsharp masking ({usm_pct:.0f}%): resta una version gaussiana '
                f'para acentuar bordes. El shock ya hace el grueso; USM es '
                f'el ultimo toque de nitidez.')

    # --- Fase 8: reconstruccion final (anti-ringing sobre display) ---
    final = anti_ringing(display, bic_display)
    agregar(final, 'Reconstruccion final',
            'Anti-ringing final contra la bicubica de display: acota el '
            'overshoot residual. Este frame es identico al resultado del '
            'boton Reconstruir.')

    return {
        'frames':          frames,
        'pasos':           pasos,
        'pasos_de_frame':  pasos_idx,
        'subtitulos':      subtitulos,
        'omega':           omega_usado,
        'iteraciones':     iters_total,
        'tiempo':          tiempo_total,
        'etapas':          etapas,
    }


def upscale_poisson_screened(tile: np.ndarray, factor: int = 4,
                              lam: float = 8.0, beta: float = 1.7,
                              tau: float = 0.02,
                              usar_back_projection: bool = True,
                              usar_anti_ringing: bool = True,
                              usar_multiescala: bool = True,
                              n_back_pass: int = 2, back_gain: float = 0.6,
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
