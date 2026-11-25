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

Amplificacion decreciente: en el ciclo k se usa
    beta_k = 1 + (beta - 1) * rho^k      (0 < rho < 1)
Si beta fuera constante, el refuerzo se acumularia ciclo tras ciclo y con
muchos ciclos la imagen se "rompe" (se midio: -6 dB con 30 ciclos). Con
beta_k decreciente el refuerzo total queda acotado y el metodo es estable
con cualquier numero de ciclos.

Se repite 'ciclos' veces y se termina con unas pasadas de back-projection.

La malla esta centrada en pixeles: la salida mide exactamente
(h*factor, w*factor), sin estirar ni recortar nada. Las regiones grandes se
procesan por bloques con solape (ver _upscale_por_bloques).

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
from core.denoise import preparar_entrada, limpiar
from core.sor_solver import (resolver_screened, resolver_screened_con_snapshots,
                             CalculoCancelado)

# Hasta este tamano de salida (pixeles) se resuelve de una sola vez.
MAX_PIXELES_BLOQUE = 4_000_000
# Por encima, se procesa por bloques; mas alla de este tope no se procesa
# (memoria y tiempo) y se pide al usuario una region mas pequena.
MAX_PIXELES_SALIDA = 25_000_000
# Bloques: lado del nucleo y solape, en pixeles de la imagen pequena.
BLOQUE_LR = 256
SOLAPE_LR = 16

# Valores por defecto ajustados con scripts/generar_resultados.py
# (4 imagenes de prueba, factores x2, x3 y x4).
PARAMS_DEFECTO = dict(lam=0.2, beta=1.8, tau=0.08, sigma=0.6, rho=0.93,
                      ciclos=10, bp_final=5, sor_iters=150, tol=1e-6)


class SalidaDemasiadoGrande(ValueError):
    """La imagen de salida superaria MAX_PIXELES_SALIDA."""


def factor_maximo(h: int, w: int) -> int:
    """Mayor factor entero cuya salida cabe en MAX_PIXELES_SALIDA (0 si ninguno)."""
    k = int(np.sqrt(MAX_PIXELES_SALIDA / max(1, h * w)))
    return k if k >= 2 else 0


def verificar_tamano(h: int, w: int, factor: int) -> None:
    """Lanza SalidaDemasiadoGrande con un mensaje claro si la salida no cabe."""
    salida = h * w * factor * factor
    if salida > MAX_PIXELES_SALIDA:
        fmax = factor_maximo(h, w)
        sugerencia = (f'Use factor {fmax} o menor, o haga zoom sobre una zona mas pequena.'
                      if fmax else 'Haga zoom sobre una zona mas pequena.')
        raise SalidaDemasiadoGrande(
            f'La region de {w}x{h} px ampliada x{factor} daria '
            f'{w * factor}x{h * factor} px ({salida / 1e6:.0f} MP); el maximo es '
            f'{MAX_PIXELES_SALIDA / 1e6:.0f} MP. {sugerencia}')


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


def beta_ciclo(beta: float, rho: float, k: int) -> float:
    """Amplificacion del ciclo k (k = 0, 1, ...): 1 + (beta - 1) * rho^k."""
    return 1.0 + (beta - 1.0) * (rho ** k)


# Pipeline de un bloque

def _rzi_canal(f_lr, factor, lam, beta, tau, sigma, rho, ciclos, bp_final,
               sor_iters, tol, omega, realce, usar_anti_ringing,
               al_paso=None, progreso=None, cancelado=None, emitir_nn=True):
    """
    Ejecuta el modelo RZI sobre un canal 2D completo (sin control de tamano).

    al_paso(frame, paso, subtitulo): se llama en cada etapa (animacion).
    progreso(fraccion, texto): avance entre 0 y 1.
    cancelado(): si devuelve True se lanza CalculoCancelado.
    """
    h, w = f_lr.shape
    H, W = h * factor, w * factor
    n_ciclos = int(ciclos)

    def paso(frame, nombre, texto):
        if al_paso is not None:
            al_paso(frame, nombre, texto)

    def revisar(frac, texto):
        if cancelado is not None and cancelado():
            raise CalculoCancelado()
        if progreso is not None:
            progreso(frac, texto)

    if al_paso is not None and emitir_nn:
        paso(resize_float(f_lr, W, H, Image.NEAREST), 'Vecino mas cercano (NN)',
             'Zoom nativo (NN): cada pixel original se replica en un bloque '
             f'de {factor}x{factor}. Sin suavizado ni nueva estructura.')

    u = np.clip(ampliar(f_lr, W, H), 0.0, 1.0)
    err0 = error_consistencia(u, f_lr)
    error_datos = [err0]
    paso(u, 'Bicubica inicial (u0)',
         f'Guia inicial bicubica u0 (tamano exacto {W}x{H}). Elimina bloques pero '
         f'suaviza bordes | error ||Down(u0)-f|| = {err0:.5f}.')

    historial = []
    iteraciones = 0
    tiempo = 0.0
    convergio = True
    omega_usado = omega

    if factor > 1:
        for c in range(1, n_ciclos + 1):
            revisar((c - 1) / (n_ciclos + 1), f'Ciclo {c} de {n_ciclos}')
            t = back_projection(u, f_lr, n_pass=1)
            b_k = beta_ciclo(beta, rho, c - 1)
            problema = construir_problema_gradiente(t, lam=lam, beta=b_k,
                                                    tau=tau, sigma=sigma)

            if c <= 2 or n_ciclos <= 5:
                nombre_c = f'Ciclo {c}: BP + SOR Red-Black'
            else:
                nombre_c = f'Ciclos 3..{n_ciclos}: refinamiento iterativo'
            if c <= 2 and al_paso is not None:
                err_bp = error_consistencia(np.clip(t, 0.0, 1.0), f_lr)
                paso(t, nombre_c,
                     f'Ciclo {c}/{n_ciclos} - Back-projection: t = u + Up(f - Down(u)) | '
                     f'||Down(t)-f|| = {err_bp:.5f}. Recupera datos pero deja ondas.')

            if c == 1 and al_paso is not None:
                cada = max(8, int(sor_iters) // 4)
                res = resolver_screened_con_snapshots(problema, omega=omega, tol=tol,
                                                      max_iter=sor_iters, cada_n_iter=cada)
                snaps = res['snapshots']
                for i in range(min(3, len(snaps) - 1)):
                    it_aprox = min(res['iteraciones'], (i + 1) * cada)
                    paso(snaps[i], nombre_c,
                         f'Ciclo 1 - SOR Red-Black (iter {it_aprox}, omega={res["omega"]:.3f}, '
                         f'beta={b_k:.2f}): (4+lam)*u - vecinos = -div(v) + lam*t.')
            else:
                res = resolver_screened(problema, omega=omega, tol=tol, max_iter=sor_iters)

            u = res['imagen_recon']
            iteraciones += res['iteraciones']
            tiempo += res['tiempo']
            convergio = convergio and res['convergio']
            omega_usado = res['omega']
            historial.extend(res['historial'].tolist())
            err_c = error_consistencia(u, f_lr)
            error_datos.append(err_c)
            paso(u, nombre_c,
                 f'Ciclo {c}/{n_ciclos} completado ({res["iteraciones"]} iters SOR, '
                 f'omega={res["omega"]:.3f}, beta={b_k:.2f}) | ||Down(u)-f|| = {err_c:.5f}.')

        revisar(n_ciclos / (n_ciclos + 1), 'Back-projection final')
        if int(bp_final) > 0:
            u = back_projection(u, f_lr, n_pass=int(bp_final))
            paso(u, 'Back-projection final',
                 f'Back-projection final ({int(bp_final)} pasadas): ajusta la consistencia '
                 f'con la imagen original | ||Down(u)-f|| = '
                 f'{error_consistencia(np.clip(u, 0.0, 1.0), f_lr):.5f}.')

        if usar_anti_ringing:
            u = anti_ringing(u, np.clip(ampliar(f_lr, W, H), 0.0, 1.0))
            paso(u, 'Anti-ringing',
                 'Recorte local 3x3 contra la guia bicubica: acota posibles halos '
                 'cerca de bordes de alto contraste.')

        u = np.clip(u, 0.0, 1.0)
        if realce > 0:
            u = realce_perceptual(u, f_lr, realce)
            paso(u, 'Realce perceptual',
                 f'Realce perceptual (fuerza={realce:.2f}): USM acotado en 3x3 + '
                 f'2 pasadas de back-projection para mayor contraste visual.')
        error_datos.append(error_consistencia(u, f_lr))

    u = np.clip(u, 0.0, 1.0)
    paso(u, 'Reconstruccion final',
         f'Resultado final RZI ({W}x{H} px) | error de consistencia ||Down(u)-f|| = '
         f'{error_datos[-1]:.5f} (desde {err0:.5f} en bicubica).')
    if progreso is not None:
        progreso(1.0, 'Listo')

    return {
        'imagen_recon': u,
        'iteraciones':  int(iteraciones),
        'tiempo':       float(tiempo),
        'omega':        float(omega_usado) if omega_usado is not None else float('nan'),
        'convergio':    bool(convergio),
        'historial':    np.asarray(historial, dtype=np.float64),
        'error_datos':  np.asarray(error_datos, dtype=np.float64),
    }


# Procesamiento por bloques

def _rampa(n: int, solape: int, inicio_libre: bool, fin_libre: bool) -> np.ndarray:
    """Peso 1D: sube linealmente en el solape de los bordes compartidos."""
    w = np.ones(n, dtype=np.float64)
    if solape > 0:
        r = (np.arange(solape) + 0.5) / solape
        if not inicio_libre:
            w[:solape] = r
        if not fin_libre:
            w[-solape:] = np.minimum(w[-solape:], r[::-1])
    return w


def _cortes(n: int, bloque: int, solape: int) -> list:
    """Intervalos [a, b) que cubren 0..n con nucleos de 'bloque' y solape."""
    if n <= bloque + 2 * solape:
        return [(0, n)]
    cortes = []
    inicio = 0
    while inicio < n:
        fin = min(n, inicio + bloque)
        cortes.append((max(0, inicio - solape), min(n, fin + solape)))
        inicio = fin
    return cortes


def _upscale_por_bloques(f_lr, factor, params, progreso=None, cancelado=None):
    """
    Amplia una imagen grande por bloques con solape y mezcla con rampas.

    La back-projection y el SOR solo usan vecinos cercanos, asi que cada
    bloque, con unos pixeles de solape, da practicamente el mismo resultado
    que resolver la imagen completa. Las rampas lineales en el solape evitan
    costuras visibles.
    """
    h, w = f_lr.shape
    H, W = h * factor, w * factor
    acumulado = np.zeros((H, W), dtype=np.float64)
    pesos = np.zeros((H, W), dtype=np.float64)
    filas = _cortes(h, BLOQUE_LR, SOLAPE_LR)
    cols = _cortes(w, BLOQUE_LR, SOLAPE_LR)
    total = len(filas) * len(cols)
    sol = SOLAPE_LR * factor

    resumen = dict(iteraciones=0, tiempo=0.0, convergio=True, omega=float('nan'))
    errores, historial = [], []
    n = 0
    for (y0, y1) in filas:
        for (x0, x1) in cols:
            base = n / total

            def prog(frac, texto, base=base):
                if progreso is not None:
                    progreso(base + frac / total, f'Bloque {n + 1} de {total}: {texto}')

            r = _rzi_canal(f_lr[y0:y1, x0:x1], factor, progreso=prog,
                           cancelado=cancelado, **params)
            wy = _rampa((y1 - y0) * factor, sol, y0 == 0, y1 == h)
            wx = _rampa((x1 - x0) * factor, sol, x0 == 0, x1 == w)
            peso = np.outer(wy, wx)
            acumulado[y0 * factor:y1 * factor, x0 * factor:x1 * factor] += peso * r['imagen_recon']
            pesos[y0 * factor:y1 * factor, x0 * factor:x1 * factor] += peso
            resumen['iteraciones'] += r['iteraciones']
            resumen['tiempo'] += r['tiempo']
            resumen['convergio'] = resumen['convergio'] and r['convergio']
            resumen['omega'] = r['omega']
            errores.append(r['error_datos'])
            if not historial:
                historial = r['historial'].tolist()
            n += 1

    u = np.clip(acumulado / np.maximum(pesos, 1e-12), 0.0, 1.0)
    resumen.update(
        imagen_recon=u,
        historial=np.asarray(historial, dtype=np.float64),
        error_datos=np.mean(np.stack(errores), axis=0),
        bloques=total,
    )
    return resumen


# API publica

def upscale_rzi(tile: np.ndarray, factor: int = 4, lam: float = 0.2,
                beta: float = 1.8, tau: float = 0.08, sigma: float = 0.6,
                rho: float = 0.93, ciclos: int = 10, bp_final: int = 5,
                sor_iters: int = 150, tol: float = 1e-6, omega: float = None,
                realce: float = 0.0, usar_anti_ringing: bool = False,
                capturar_snapshots: bool = False, al_paso=None,
                progreso=None, cancelado=None, limpieza: str = 'auto',
                sigma_ruido: float = None) -> dict:
    """
    Amplia una imagen en escala de grises 'factor' veces con el modelo RZI.

    Parametros principales
    ----------------------
    tile      : ndarray float64 [0,1], forma (h, w).
    factor    : factor de ampliacion entero (>= 1).
    lam       : peso de fidelidad del Poisson screened.
    beta, tau : amplificacion de bordes del primer ciclo (beta=1 la desactiva).
    rho       : decaimiento de la amplificacion por ciclo (0 < rho <= 1).
    ciclos    : cuantas veces se alterna back-projection + SOR.
    bp_final  : pasadas finales de back-projection.
    omega     : None usa el omega optimo de la malla; un valor lo fuerza.
    realce    : 0..1, realce perceptual opcional (ver realce_perceptual).
    al_paso, progreso, cancelado : callbacks opcionales (ver _rzi_canal).
    limpieza  : 'auto' (limpia solo si detecta ruido claro), 'ruido', 'jpeg'
                o 'no'. Ver core/denoise.py.
    sigma_ruido : ruido medido sobre la imagen completa (opcional; mas
                confiable que medirlo en una region pequena).

    Las regiones cuya salida supera MAX_PIXELES_BLOQUE se procesan por
    bloques; las que superan MAX_PIXELES_SALIDA lanzan SalidaDemasiadoGrande.

    Retorna dict con:
        'imagen_recon' : ndarray (h*factor, w*factor) en [0,1].
        'iteraciones', 'tiempo', 'omega', 'convergio', 'historial',
        'error_datos'  : resumen del solver.
        'snapshots'    : estados intermedios si capturar_snapshots (o None).
        'por_bloques'  : True si se proceso por bloques.
        'sigma_estimado', 'limpieza_aplicada' : ruido medido y limpieza usada.
        'tile_reducido': siempre False (se conserva por compatibilidad).
    """
    if tile.ndim != 2:
        raise ValueError("upscale_rzi espera un canal 2D; use upscale() para color.")
    factor = int(factor)
    if factor < 1:
        raise ValueError("El factor debe ser >= 1.")
    f_original = np.asarray(tile, dtype=np.float64)
    h, w = f_original.shape
    verificar_tamano(h, w, factor)
    f_lr, sigma_est, aplicada = preparar_entrada(f_original, limpieza, sigma_ruido)

    params = dict(lam=lam, beta=beta, tau=tau, sigma=sigma, rho=rho, ciclos=ciclos,
                  bp_final=bp_final, sor_iters=sor_iters, tol=tol, omega=omega,
                  realce=realce, usar_anti_ringing=usar_anti_ringing)

    grande = h * w * factor * factor > MAX_PIXELES_BLOQUE
    if grande and (al_paso is not None or capturar_snapshots):
        raise SalidaDemasiadoGrande(
            'La animacion paso a paso necesita una region mas pequena '
            f'(salida de hasta {MAX_PIXELES_BLOQUE / 1e6:.0f} MP). Haga zoom.')

    snapshots = None
    if grande:
        res = _upscale_por_bloques(f_lr, factor, params, progreso, cancelado)
    else:
        if capturar_snapshots:
            snapshots = []
            previo = al_paso

            def al_paso(frame, nombre, texto, _previo=previo):
                if not nombre.startswith(('Vecino', 'Entrada', 'Limpieza')):
                    snapshots.append(np.clip(frame, 0.0, 1.0))
                if _previo is not None:
                    _previo(frame, nombre, texto)

        emitir_nn = True
        if aplicada != 'no' and al_paso is not None:
            H, W = h * factor, w * factor
            al_paso(resize_float(f_original, W, H, Image.NEAREST), 'Entrada original',
                    f'Imagen de entrada vista con zoom nativo. Ruido estimado en zonas planas: '
                    f'sigma = {sigma_est:.4f}.')
            texto = ('ruido' if aplicada == 'ruido' else 'bloques de JPEG')
            al_paso(resize_float(f_lr, W, H, Image.NEAREST),
                    'Limpieza de la entrada (variacion total)',
                    f'Se limpia el {texto} con variacion total (modelo ROF): aplana el ruido '
                    f'y conserva los bordes. Sin esto, la back-projection copiaria el ruido.')
            emitir_nn = False
        res = _rzi_canal(f_lr, factor, al_paso=al_paso, progreso=progreso,
                         cancelado=cancelado, emitir_nn=emitir_nn, **params)

    res.update(snapshots=snapshots, por_bloques=grande, tile_reducido=False,
               factor=factor, lam=float(lam), sigma_estimado=sigma_est,
               limpieza_aplicada=aplicada)
    return res


def _upscale_croma(c: np.ndarray, factor: int, n_bp: int = 3) -> np.ndarray:
    """Crominancia: bicubica + back-projection (barato, sin regularizacion)."""
    h, w = c.shape
    up = ampliar(c, w * factor, h * factor)
    return np.clip(back_projection(up, c, n_pass=n_bp), 0.0, 1.0)


def _limpiar_croma(cb, cr, res) -> tuple:
    """Aplica a Cb y Cr la misma limpieza que se uso en la luminancia."""
    modo = res.get('limpieza_aplicada', 'no')
    if modo == 'no':
        return cb, cr
    sigma = res.get('sigma_estimado', 0.0)
    return limpiar(cb, modo, sigma), limpiar(cr, modo, sigma)


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

    verificar_tamano(tile.shape[0], tile.shape[1], int(factor))
    y, cb, cr = rgb_a_ycbcr(tile[..., :3])
    res = upscale_rzi(y, factor, **kwargs)
    cb, cr = _limpiar_croma(cb, cr, res)
    cb_up = _upscale_croma(cb, factor)
    cr_up = _upscale_croma(cr, factor)

    res['imagen_recon'] = ycbcr_a_rgb(res['imagen_recon'], cb_up, cr_up)
    if res['snapshots'] is not None:
        res['snapshots'] = [ycbcr_a_rgb(s, cb_up, cr_up) for s in res['snapshots']]
    return res


# Animacion didactica paso a paso

def generar_frames_pipeline(tile: np.ndarray, factor: int = 4, progreso=None,
                            cancelado=None, **kwargs) -> dict:
    """
    Genera la secuencia didactica completa paso a paso para "Animar convergencia".

    Usa exactamente el mismo pipeline que upscale() (callback al_paso), asi
    que el ultimo frame es identico al resultado de upscale() con los mismos
    parametros. Acepta imagen en grises (h, w) o RGB (h, w, 3).

    Retorna dict con 'frames', 'pasos', 'pasos_de_frame', 'subtitulos',
    'bicubica' y el resumen del solver ('omega', 'iteraciones', 'tiempo',
    'convergio', 'historial', 'error_datos', 'ciclos', 'factor').
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("El factor debe ser >= 1.")
    arr = np.asarray(tile, dtype=np.float64)
    if arr.ndim == 3 and (arr.shape[2] < 3):
        raise ValueError("Se espera imagen 2D o RGB (h, w, 3).")

    frames, pasos, pasos_idx, subtitulos = [], [], [], []

    def al_paso(frame, paso, subtitulo):
        if paso not in pasos:
            pasos.append(paso)
        pasos_idx.append(pasos.index(paso))
        frames.append(np.clip(np.asarray(frame, dtype=np.float64), 0.0, 1.0))
        subtitulos.append(subtitulo)

    if arr.ndim == 2:
        res = upscale_rzi(arr, factor, al_paso=al_paso, progreso=progreso,
                          cancelado=cancelado, **kwargs)
        h, w = arr.shape
        bicubica = np.clip(resize_float(arr, w * factor, h * factor, Image.BICUBIC), 0, 1)
    else:
        rgb = arr[..., :3]
        verificar_tamano(rgb.shape[0], rgb.shape[1], factor)
        y, cb, cr = rgb_a_ycbcr(rgb)
        res = upscale_rzi(y, factor, al_paso=al_paso, progreso=progreso,
                          cancelado=cancelado, **kwargs)
        h, w = y.shape
        H, W = h * factor, w * factor
        cb_l, cr_l = _limpiar_croma(cb, cr, res)
        cb_up = _upscale_croma(cb_l, factor)
        cr_up = _upscale_croma(cr_l, factor)
        frames = [ycbcr_a_rgb(fy, cb_up, cr_up) for fy in frames]
        # los cuadros de la entrada original (vecino y bicubica) se hacen
        # directamente en RGB para que la comparacion visual sea fiel
        nn_rgb = resize_rgb_float(rgb, W, H, Image.NEAREST)
        bic_rgb = resize_rgb_float(rgb, W, H, Image.BICUBIC)
        sin_limpieza = res['limpieza_aplicada'] == 'no'
        for i, pi in enumerate(pasos_idx):
            nombre = pasos[pi]
            if nombre == 'Entrada original' or (sin_limpieza and nombre.startswith('Vecino')):
                frames[i] = nn_rgb
            elif sin_limpieza and nombre.startswith('Bicubica inicial'):
                frames[i] = bic_rgb
        bicubica = bic_rgb

    return {
        'frames':         frames,
        'pasos':          pasos,
        'pasos_de_frame': pasos_idx,
        'subtitulos':     subtitulos,
        'bicubica':       bicubica,
        'omega':          res['omega'],
        'iteraciones':    res['iteraciones'],
        'tiempo':         res['tiempo'],
        'convergio':      res['convergio'],
        'historial':      res['historial'],
        'error_datos':    res['error_datos'],
        'ciclos':         int(kwargs.get('ciclos', PARAMS_DEFECTO['ciclos'])),
        'factor':         factor,
        'tile_reducido':  False,
        'sigma_estimado': res['sigma_estimado'],
        'limpieza_aplicada': res['limpieza_aplicada'],
    }
