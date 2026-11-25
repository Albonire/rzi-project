"""
Capa 3 - Diagnostico del entorno y de cada herramienta de la aplicacion.

Ejecuta, sin interfaz grafica, exactamente los mismos pasos que hacen
"Medir calidad", "Barrido de omega", "Benchmark" y "Ampliar" (incluido dibujar
la grafica a imagen) e informa versiones, tiempos y cualquier error con su
traceback completo. Se usa desde Ayuda > Diagnostico y desde
scripts/diagnostico.py.
"""

import platform
import sys
import time
import traceback

import numpy as np


def _version(modulo: str) -> str:
    try:
        m = __import__(modulo)
        return getattr(m, '__version__', 'instalado')
    except Exception as exc:  # el diagnostico nunca debe caerse
        return f'NO DISPONIBLE ({type(exc).__name__}: {exc})'


def _imagen_prueba(n: int = 192) -> np.ndarray:
    """Imagen sintetica con bordes y texturas (no depende de scikit-image)."""
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.5 + 0.3 * np.sin(18 * x) * np.cos(11 * y)
    img[n // 4:n // 2, n // 4:3 * n // 4] = 0.9
    img[(x - 0.7) ** 2 + (y - 0.7) ** 2 < 0.02] = 0.1
    return np.clip(img, 0, 1)


def ejecutar_diagnostico(progreso=None) -> tuple:
    """
    Retorna (texto_del_informe, todo_ok). 'progreso(fraccion, texto)' es opcional.
    """
    lineas = []
    ok_total = True

    def log(t=''):
        lineas.append(t)

    log('DIAGNOSTICO RZI')
    log(f'Sistema: {platform.platform()}')
    log(f'Python: {sys.version.split()[0]} ({sys.executable})')
    for mod in ('numpy', 'scipy', 'PIL', 'matplotlib', 'numba', 'skimage', 'PySide6'):
        log(f'  {mod:<11} {_version(mod)}')
    try:
        import matplotlib
        log(f'  backend de matplotlib: {matplotlib.get_backend()}')
    except Exception:
        pass

    from core.sor_solver import NUMBA_DISPONIBLE
    log(f'Numba activo: {"SI" if NUMBA_DISPONIBLE else "NO (todo sera muy lento)"}')
    if NUMBA_DISPONIBLE:
        try:
            import numba
            log(f'  capa de hilos: {numba.config.THREADING_LAYER}, hilos: {numba.config.NUMBA_NUM_THREADS}')
        except Exception:
            pass
    log()

    pasos = []

    def paso(nombre):
        def deco(fn):
            pasos.append((nombre, fn))
            return fn
        return deco

    @paso('Compilar kernels (Numba)')
    def _p1():
        from core.sor_solver import _calentamiento
        _calentamiento()
        return 'ok'

    @paso('Medir calidad (degradar, ampliar, metricas, grafica)')
    def _p2():
        from core.upscaler import upscale
        from services.metrics import (degradar, interpolacion_bicubica,
                                      interpolacion_lanczos, tabla_comparativa)
        from services.exporter import grafica_comparacion, elegir_recorte, figura_a_imagen
        hr, lr = degradar(_imagen_prueba(), 4)
        cand = {'Bicubica': interpolacion_bicubica(lr, 4),
                'Lanczos': interpolacion_lanczos(lr, 4),
                'RZI (fiel)': upscale(lr, 4)['imagen_recon']}
        tabla = tabla_comparativa(hr, cand, modelos_rzi=('RZI (fiel)',))
        img = figura_a_imagen(grafica_comparacion(hr, cand, tabla,
                                                  recorte=elegir_recorte(hr, 64)))
        g = tabla[2]['psnr'] - tabla[0]['psnr']
        return f'RZI {g:+.2f} dB sobre bicubica, figura {img.shape[1]}x{img.shape[0]} px'

    @paso('Barrido de omega (problema, barrido, grafica)')
    def _p3():
        from core.poisson_builder import construir_problema
        from core.sor_solver import barrido_omega
        from services.exporter import grafica_barrido_omega, figura_a_imagen
        p = construir_problema(_imagen_prueba(128), 4)
        r = barrido_omega(p, tol=1e-3, max_iter=300, n_puntos=15)
        img = figura_a_imagen(grafica_barrido_omega(r['omegas'], r['iteraciones'],
                                                    r['omega_teorico'],
                                                    r['omega_malla_completa']))
        return (f"omega empirico {r['omega_empirico']:.3f}, estimado {r['omega_teorico']:.3f}, "
                f'figura {img.shape[1]}x{img.shape[0]} px')

    @paso('Benchmark Numba vs Python (grafica)')
    def _p4():
        from core.poisson_builder import construir_problema
        from core.sor_solver import benchmark_numba_vs_python
        from services.exporter import grafica_benchmark, figura_a_imagen
        r = benchmark_numba_vs_python(construir_problema(_imagen_prueba(64), 4), n_iter=5)
        img = figura_a_imagen(grafica_benchmark([r]))
        return f"aceleracion x{r['aceleracion']:.0f}, figura {img.shape[1]}x{img.shape[0]} px"

    @paso('Ampliar por bloques (region grande)')
    def _p5():
        from core.upscaler import upscale_rzi
        tile = np.tile(_imagen_prueba(160), (2, 2))
        r = upscale_rzi(tile, 7, ciclos=2, sor_iters=40)
        H, W = r['imagen_recon'].shape
        return f"{tile.shape[1]}x{tile.shape[0]} -> {W}x{H} px, bloques={r.get('bloques', 1)}"

    for k, (nombre, fn) in enumerate(pasos):
        if progreso is not None:
            progreso(k / len(pasos), nombre)
        t0 = time.perf_counter()
        try:
            detalle = fn()
            log(f'[OK]    {nombre} ({time.perf_counter() - t0:.1f} s): {detalle}')
        except Exception:
            ok_total = False
            log(f'[ERROR] {nombre} ({time.perf_counter() - t0:.1f} s)')
            log(traceback.format_exc())
    if progreso is not None:
        progreso(1.0, 'Listo')

    log()
    log('Resultado: todo funciona.' if ok_total else
        'Resultado: hay errores. Copie este informe y envielo al equipo.')
    return '\n'.join(lineas), ok_total
