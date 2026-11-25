"""
Capa 2 - Modelo RZI de ampliacion (upscaling) de imagenes.

Idea en una frase: buscamos una imagen grande u que (1) al reducirla vuelva
a dar la imagen pequena original y (2) tenga bordes firmes. Lo logramos
alternando dos pasos:

  a) Back-projection (consistencia con los datos):
         t = u + Up( f - Down(u) )
     Down = reduccion bicubica con antialias (modelo de como se "achica"
     una imagen), Up = ampliacion bicubica. Si Down(u) ya coincide con f,
     la correccion es cero.

  b) Poisson "screened" con gradientes amplificados (regularizacion), que
     se resuelve con SOR Red-Black + Numba:
         (4 + lam) u - (suma de vecinos) = -div(v) + lam * t
     v es el campo de gradientes de t, reforzado en los bordes reales
     (ver core.poisson_builder.amplified_guidance). La ecuacion limpia el
     "ringing" que deja la back-projection y mantiene los bordes nitidos.

Se repite 'ciclos' veces y se termina con unas pasadas de back-projection.

La malla esta centrada en pixeles: la salida mide exactamente
(h*factor, w*factor), sin estirar ni recortar nada.

Imagenes a color: se procesa la luminancia Y (donde el ojo percibe la
nitidez) con el modelo completo, y la crominancia (Cb, Cr) con bicubica mas
back-projection.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter

from core.image_ops import (resize_float, resize_rgb_float,
                            unsharp_mask_float, rgb_a_ycbcr, ycbcr_a_rgb)
from core.poisson_builder import construir_problema_gradiente
from core.sor_solver import resolver_screened, resolver_screened_con_snapshots

# Tope de tamano de salida (pixeles) para no saturar memoria ni tiempo.
MAX_PIXELES_SALIDA = 4_000_000

# Valores por defecto ajustados con scripts/generar_resultados.py
# (4 imagenes de prueba, factores x2, x3 y x4).
PARAMS_DEFECTO = dict(lam=0.3, beta=1.6, tau=0.05, sigma=0.6, ciclos=10,
                      bp_final=5, sor_iters=150, tol=1e-6)


def reducir(u: np.ndarray, w: int, h: int) -> np.ndarray:
    """Operador Down: reduccion bicubica con antialias (PIL escala el nucleo)."""
    return resize_float(u, w, h, Image.BICUBIC)


def ampliar(u: np.ndarray, w: int, h: int) -> np.ndarray:
    """Operador Up: ampliacion bicubica."""
    return resize_float(u, w, h, Image.BICUBIC)


def back_projection(u: np.ndarray, f_lr: np.ndarray, n_pass: int = 1,
                    gain: float = 1.0) -> np.ndarray:
    """
    Back-projection iterativa (Irani y Peleg, 1991):
        u <- u + gain * Up( f_lr - Down(u) )
    Cada pasada reduce el error de consistencia ||Down(u) - f_lr||.
    """
    h_lr, w_lr = f_lr.shape
    h_hi, w_hi = u.shape
    out = u
    for _ in range(n_pass):
        residuo = f_lr - reducir(out, w_lr, h_lr)
        out = out + gain * ampliar(residuo, w_hi, h_hi)
    return out


def error_consistencia(u: np.ndarray, f_lr: np.ndarray) -> float:
    """Error RMS entre Down(u) y la imagen pequena original."""
    h, w = f_lr.shape
    return float(np.sqrt(np.mean((reducir(u, w, h) - f_lr) ** 2)))


def anti_ringing(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Recorta u al rango [min, max] local 3x3 de una referencia."""
    return np.clip(u, minimum_filter(ref, size=3), maximum_filter(ref, size=3))


def realce_perceptual(u: np.ndarray, f_lr: np.ndarray, fuerza: float) -> np.ndarray:
    """
    Realce opcional de nitidez visual (por defecto apagado).

    Unsharp mask controlado: se suma 1.5*fuerza veces el detalle fino
    (u - Gauss(u)), se recorta al rango local 3x3 de u (evita halos) y se
    reproyecta 2 veces para no perder la consistencia con los datos.
    Hace los bordes mas marcados a la vista, a cambio de bajar un poco el
    PSNR (es un realce "para el ojo", no una mejora de fidelidad).
    """
    if fuerza <= 0:
        return u
    v = unsharp_mask_float(u, percent=150.0 * fuerza, radius=1.2)
    v = anti_ringing(v, u)
    return np.clip(back_projection(v, f_lr, n_pass=2), 0.0, 1.0)


def _ajustar_tamano(tile: np.ndarray, factor: int) -> tuple:
    """Si la salida superaria MAX_PIXELES_SALIDA, reduce el tile de entrada."""
    h, w = tile.shape[:2]
    if h * w * factor * factor <= MAX_PIXELES_SALIDA:
        return tile, False
    escala = np.sqrt(MAX_PIXELES_SALIDA / (h * w * factor * factor))
    nw, nh = max(8, int(w * escala)), max(8, int(h * escala))
    if tile.ndim == 2:
        return np.clip(resize_float(tile, nw, nh, Image.LANCZOS), 0, 1), True
    canales = [resize_float(tile[..., c], nw, nh, Image.LANCZOS) for c in range(3)]
    return np.clip(np.stack(canales, axis=-1), 0, 1), True


def upscale_rzi(tile: np.ndarray, factor: int = 4, lam: float = 0.3,
                beta: float = 1.6, tau: float = 0.05, sigma: float = 0.6,
                ciclos: int = 10, bp_final: int = 5, sor_iters: int = 150,
                tol: float = 1e-6, omega: float = None, realce: float = 0.0,
                usar_anti_ringing: bool = False,
                capturar_snapshots: bool = False) -> dict:
    """
    Amplia una imagen en escala de grises 'factor' veces con el modelo RZI.

    Parametros principales
    ----------------------
    tile      : ndarray float64 [0,1], forma (h, w).
    factor    : factor de ampliacion entero (>= 1).
    lam       : peso de fidelidad del Poisson screened.
    beta, tau : amplificacion de bordes (beta=1 desactiva la amplificacion).
    ciclos    : cuantas veces se alterna back-projection + SOR.
    bp_final  : pasadas finales de back-projection.
    omega     : None usa el omega optimo de la malla; un valor lo fuerza.
    realce    : 0..1, realce perceptual opcional (ver realce_perceptual).

    Retorna dict con:
        'imagen_recon'    : ndarray (h*factor, w*factor) en [0,1].
        'iteraciones'     : total de iteraciones SOR.
        'tiempo'          : segundos dentro del solver SOR.
        'omega'           : omega usado.
        'convergio'       : True si todos los SOR alcanzaron la tolerancia.
        'historial'       : residuo relativo de todas las iteraciones SOR.
        'error_datos'     : error RMS ||Down(u) - f|| tras cada ciclo.
        'snapshots'       : estados intermedios (o None).
        'tile_reducido'   : True si se redujo la entrada por el tope de memoria.
    """
    if tile.ndim != 2:
        raise ValueError("upscale_rzi espera un canal 2D; use upscale() para color.")
    factor = int(factor)
    if factor < 1:
        raise ValueError("El factor debe ser >= 1.")

    f_lr, reducido = _ajustar_tamano(np.asarray(tile, dtype=np.float64), factor)
    h, w = f_lr.shape
    H, W = h * factor, w * factor

    u = np.clip(ampliar(f_lr, W, H), 0.0, 1.0)
    snapshots = [u.copy()] if capturar_snapshots else None
    error_datos = [error_consistencia(u, f_lr)]
    historial = []
    iteraciones = 0
    tiempo = 0.0
    convergio = True
    omega_usado = omega

    if factor > 1:
        for _ in range(int(ciclos)):
            t = back_projection(u, f_lr, n_pass=1)
            problema = construir_problema_gradiente(t, lam=lam, beta=beta,
                                                    tau=tau, sigma=sigma)
            res = resolver_screened(problema, omega=omega, tol=tol,
                                    max_iter=sor_iters)
            u = res['imagen_recon']
            iteraciones += res['iteraciones']
            tiempo += res['tiempo']
            convergio = convergio and res['convergio']
            omega_usado = res['omega']
            historial.extend(res['historial'].tolist())
            error_datos.append(error_consistencia(u, f_lr))
            if capturar_snapshots:
                snapshots.append(np.clip(u, 0.0, 1.0))

        u = back_projection(u, f_lr, n_pass=int(bp_final))

        if usar_anti_ringing:
            u = anti_ringing(u, np.clip(ampliar(f_lr, W, H), 0.0, 1.0))
        u = np.clip(u, 0.0, 1.0)
        u = realce_perceptual(u, f_lr, realce)
        error_datos.append(error_consistencia(u, f_lr))

    u = np.clip(u, 0.0, 1.0)
    if capturar_snapshots:
        snapshots.append(u.copy())

    return {
        'imagen_recon':  u,
        'iteraciones':   int(iteraciones),
        'tiempo':        float(tiempo),
        'omega':         float(omega_usado) if omega_usado is not None else float('nan'),
        'convergio':     bool(convergio),
        'historial':     np.asarray(historial, dtype=np.float64),
        'error_datos':   np.asarray(error_datos, dtype=np.float64),
        'snapshots':     snapshots,
        'tile_reducido': reducido,
        'factor':        factor,
        'lam':           float(lam),
    }


def _upscale_croma(c: np.ndarray, factor: int, n_bp: int = 3) -> np.ndarray:
    """Crominancia: bicubica + back-projection (barato, sin regularizacion)."""
    h, w = c.shape
    up = ampliar(c, w * factor, h * factor)
    return np.clip(back_projection(up, c, n_pass=n_bp), 0.0, 1.0)


def upscale(tile: np.ndarray, factor: int = 4, **kwargs) -> dict:
    """
    Punto de entrada general: acepta imagen en grises (h, w) o RGB (h, w, 3).

    En color, la luminancia Y pasa por upscale_rzi y Cb/Cr por bicubica con
    back-projection; luego se recombinan en RGB. El dict devuelto es el de
    upscale_rzi, con 'imagen_recon' (y 'snapshots') en RGB si la entrada lo era.
    """
    tile = np.asarray(tile, dtype=np.float64)
    if tile.ndim == 2:
        return upscale_rzi(tile, factor, **kwargs)
    if tile.ndim != 3 or tile.shape[2] < 3:
        raise ValueError("Se espera imagen 2D o RGB (h, w, 3).")

    tile, reducido = _ajustar_tamano(tile[..., :3], factor)
    y, cb, cr = rgb_a_ycbcr(tile)
    res = upscale_rzi(y, factor, **kwargs)
    res['tile_reducido'] = reducido
    cb_up = _upscale_croma(cb, factor)
    cr_up = _upscale_croma(cr, factor)

    res['imagen_recon'] = ycbcr_a_rgb(res['imagen_recon'], cb_up, cr_up)
    if res['snapshots'] is not None:
        res['snapshots'] = [ycbcr_a_rgb(s, cb_up, cr_up) for s in res['snapshots']]
    return res


def _seleccionar_indices(n: int, maximo: int) -> list:
    """Submuestrea indices conservando primero y ultimo."""
    if n <= maximo:
        return list(range(n))
    sel = np.linspace(0, n - 1, maximo).round().astype(int)
    return sorted(set(sel.tolist()))


def _generar_frames_canal(canal: np.ndarray, factor: int, lam: float, beta: float,
                          tau: float, sigma: float, ciclos: int, bp_final: int,
                          sor_iters: int, tol: float, omega: float,
                          realce: float, usar_anti_ringing: bool) -> dict:
    """Ejecuta el pipeline RZI sobre un canal 2D registrando pasos y subtitulos."""
    f_lr = np.asarray(canal, dtype=np.float64)
    h, w = f_lr.shape
    H, W = h * factor, w * factor

    frames, pasos, pasos_idx, subtitulos = [], [], [], []

    def agregar(frame, paso, subtitulo):
        if paso not in pasos:
            pasos.append(paso)
        pasos_idx.append(pasos.index(paso))
        frames.append(np.clip(frame.astype(np.float64), 0.0, 1.0))
        subtitulos.append(subtitulo)

    # Fase 1: vecino mas cercano
    nn = resize_float(f_lr, W, H, Image.NEAREST)
    agregar(nn, 'Vecino mas cercano (NN)',
            'Zoom nativo (NN): cada pixel original se replica en un bloque '
            f'de {factor}x{factor}. Sin suavizado ni nueva estructura.')

    # Fase 2: bicubica inicial (u0)
    u = np.clip(ampliar(f_lr, W, H), 0.0, 1.0)
    err0 = error_consistencia(u, f_lr)
    error_datos = [err0]
    agregar(u, 'Bicubica inicial (u0)',
            f'Guia inicial bicubica u0 (tamano exacto {W}x{H}). Elimina '
            f'bloques pero suaviza bordes | error ||Down(u0)-f|| = {err0:.5f}.')

    iteraciones = 0
    tiempo = 0.0
    convergio = True
    omega_usado = omega
    historial = []
    n_ciclos = int(ciclos)

    if factor > 1:
        cada_snap = max(8, int(sor_iters) // 4)
        for c in range(1, n_ciclos + 1):
            t = back_projection(u, f_lr, n_pass=1)
            err_bp = error_consistencia(np.clip(t, 0.0, 1.0), f_lr)
            problema = construir_problema_gradiente(t, lam=lam, beta=beta,
                                                    tau=tau, sigma=sigma)

            if c <= 2:
                paso_c = f'Ciclo {c}: BP + SOR Red-Black'
                agregar(t, paso_c,
                        f'Ciclo {c}/{n_ciclos} - Back-projection: t = u + Up(f - Down(u)) | '
                        f'||Down(t)-f|| = {err_bp:.5f}. Recupera datos pero deja ondas.')
            elif n_ciclos <= 5:
                paso_c = f'Ciclo {c}: BP + SOR Red-Black'
            else:
                paso_c = f'Ciclos 3..{n_ciclos}: refinamiento iterativo'

            if c == 1:
                res = resolver_screened_con_snapshots(
                    problema, omega=omega, tol=tol, max_iter=sor_iters,
                    cada_n_iter=cada_snap)
                snaps = res['snapshots']
                sel = _seleccionar_indices(len(snaps), 4)
                omega_et = res['omega']
                for idx_s in sel[:-1]:
                    it_aprox = min(res['iteraciones'], (idx_s + 1) * cada_snap)
                    agregar(snaps[idx_s], paso_c,
                            f'Ciclo 1 - SOR Red-Black (iter {it_aprox}, omega={omega_et:.3f}): '
                            f'(4+lam)*u - vecinos = -div(v) + lam*t.')
            else:
                res = resolver_screened(problema, omega=omega, tol=tol,
                                        max_iter=sor_iters)

            u = res['imagen_recon']
            iteraciones += res['iteraciones']
            tiempo += res['tiempo']
            convergio = convergio and res['convergio']
            omega_usado = res['omega']
            historial.extend(res['historial'].tolist())

            err_c = error_consistencia(u, f_lr)
            error_datos.append(err_c)
            agregar(u, paso_c,
                    f'Ciclo {c}/{n_ciclos} completado ({res["iteraciones"]} iters SOR, '
                    f'omega={res["omega"]:.3f}) | ||Down(u)-f|| = {err_c:.5f}.')

        if int(bp_final) > 0:
            u = back_projection(u, f_lr, n_pass=int(bp_final))
            err_bpf = error_consistencia(np.clip(u, 0.0, 1.0), f_lr)
            agregar(u, 'Back-projection final',
                    f'Back-projection final ({int(bp_final)} pasadas): ajusta la '
                    f'consistencia con la imagen original | ||Down(u)-f|| = {err_bpf:.5f}.')

        if usar_anti_ringing:
            u = anti_ringing(u, np.clip(ampliar(f_lr, W, H), 0.0, 1.0))
            agregar(u, 'Anti-ringing',
                    'Recorte local 3x3 contra la guia bicubica: acota posibles halos '
                    'cerca de bordes de alto contraste.')

        u = np.clip(u, 0.0, 1.0)
        if realce > 0:
            u = realce_perceptual(u, f_lr, realce)
            agregar(u, 'Realce perceptual',
                    f'Realce perceptual (fuerza={realce:.2f}): USM acotado en 3x3 + '
                    f'2 pasadas de back-projection para mayor contraste visual.')

        err_fin = error_consistencia(u, f_lr)
        error_datos.append(err_fin)
    else:
        err_fin = err0

    u = np.clip(u, 0.0, 1.0)
    agregar(u, 'Reconstruccion final',
            f'Resultado final RZI ({W}x{H} px) | error de consistencia '
            f'||Down(u)-f|| = {err_fin:.5f} (desde {err0:.5f} en bicubica).')

    return {
        'frames':        frames,
        'pasos':         pasos,
        'pasos_de_frame': pasos_idx,
        'subtitulos':    subtitulos,
        'omega':         float(omega_usado) if omega_usado is not None else float('nan'),
        'iteraciones':   int(iteraciones),
        'tiempo':        float(tiempo),
        'convergio':     bool(convergio),
        'historial':     np.asarray(historial, dtype=np.float64),
        'error_datos':   np.asarray(error_datos, dtype=np.float64),
        'ciclos':        n_ciclos,
        'factor':        factor,
    }


def generar_frames_pipeline(tile: np.ndarray, factor: int = 4, lam: float = 0.3,
                            beta: float = 1.6, tau: float = 0.05, sigma: float = 0.6,
                            ciclos: int = 10, bp_final: int = 5, sor_iters: int = 150,
                            tol: float = 1e-6, omega: float = None,
                            realce: float = 0.0, usar_anti_ringing: bool = False) -> dict:
    """
    Genera la secuencia didactica completa paso a paso para "Animar convergencia".

    Acepta imagen en escala de grises (h, w) o RGB (h, w, 3). El ultimo frame
    es identico al resultado de upscale() con los mismos parametros.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("El factor debe ser >= 1.")

    arr = np.asarray(tile, dtype=np.float64)
    if arr.ndim == 2:
        f_lr, reducido = _ajustar_tamano(arr, factor)
        gen = _generar_frames_canal(
            f_lr, factor, lam, beta, tau, sigma, ciclos, bp_final,
            sor_iters, tol, omega, realce, usar_anti_ringing)
        gen['tile_reducido'] = reducido
        gen['bicubica'] = gen['frames'][1]
        return gen

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Se espera imagen 2D o RGB (h, w, 3).")

    rgb_lr, reducido = _ajustar_tamano(arr[..., :3], factor)
    y, cb, cr = rgb_a_ycbcr(rgb_lr)
    gen = _generar_frames_canal(
        y, factor, lam, beta, tau, sigma, ciclos, bp_final,
        sor_iters, tol, omega, realce, usar_anti_ringing)

    h, w = y.shape
    H, W = h * factor, w * factor
    cb_up = _upscale_croma(cb, factor)
    cr_up = _upscale_croma(cr, factor)

    frames_rgb = [ycbcr_a_rgb(fy, cb_up, cr_up) for fy in gen['frames']]
    # keep first NN and bicubic frames purely spatial on RGB so comparison is faithful
    frames_rgb[0] = resize_rgb_float(rgb_lr, W, H, Image.NEAREST)
    bic_rgb = resize_rgb_float(rgb_lr, W, H, Image.BICUBIC)
    frames_rgb[1] = bic_rgb

    gen['frames'] = frames_rgb
    gen['bicubica'] = bic_rgb
    gen['tile_reducido'] = reducido
    return gen
