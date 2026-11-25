"""
Capa 2 - Solver SOR Red-Black acelerado con Numba.

SOR (Successive Over-Relaxation) actualiza cada incognita con
    u_nuevo = u + omega * (u_GaussSeidel - u)
Con omega = 1 es Gauss-Seidel; con 1 < omega < 2 se "sobre-relaja" y, con el
omega adecuado, converge muchisimo mas rapido.

Enfoque matrix-free: la matriz A del sistema nunca se construye. El estencil
de 5 puntos se aplica directamente sobre la malla 2D de pixeles, asi que la
memoria usada es solo la de las imagenes (u, b, mascara).

Ordenamiento Rojo-Negro: la malla se pinta como un tablero de ajedrez. Un
pixel rojo ((i+j) par) solo tiene vecinos negros y viceversa. Por eso todos
los rojos se pueden actualizar a la vez (en paralelo) y despues todos los
negros, sin condiciones de carrera.
"""

import threading
import time

import numpy as np

try:
    from numba import njit, prange
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    prange = range

    def njit(*args, **kwargs):
        """Decorador vacio cuando Numba no esta instalado."""
        def decorador(f):
            return f
        return decorador


# La capa de hilos por defecto de Numba ("workqueue") no admite llamadas
# simultaneas a funciones paralelas desde dos hilos de Python. La interfaz
# puede lanzar la lupa y una reconstruccion a la vez, asi que los kernels se
# ejecutan de a uno.
_LOCK_KERNEL = threading.Lock()


@njit(cache=True, fastmath=True)
def _kernel_sor_rb(u, b, mascara, omega, max_iter, tol):
    """
    SOR Red-Black para 4*u - (vecinos) = b con pixeles fijos (mascara=True).

    tol es sobre la norma L2 ABSOLUTA del residuo en los pixeles libres.
    Retorna (u, iteraciones, convergio, historial_residuo).
    """
    nrows, ncols = u.shape
    historial = np.empty(max_iter, dtype=np.float64)
    iteraciones = max_iter
    convergio = False

    for k in range(max_iter):
        # Fase Roja: (i+j) par
        for i in range(1, nrows - 1):
            j_inicio = 2 - (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]
                            + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])

        # Fase Negra: (i+j) impar
        for i in range(1, nrows - 1):
            j_inicio = 1 + (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]
                            + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])

        # Residuo r = b - A u
        res2 = 0.0
        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                if not mascara[i, j]:
                    r = (b[i, j] - 4.0 * u[i, j]
                         + u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    res2 += r * r
        historial[k] = np.sqrt(res2)

        if historial[k] < tol:
            iteraciones = k + 1
            convergio = True
            break

    return u, iteraciones, convergio, historial[:iteraciones]


@njit(cache=True, fastmath=True, parallel=True)
def _kernel_sor_rb_screened(u, b, lam_map, omega, max_iter, tol):
    """
    SOR Red-Black paralelo para la ecuacion "screened":

        (4 + lam_map[i,j]) * u[i,j] - (suma de vecinos) = b[i,j]

    Los bordes (fila/columna 0 y ultima) no se actualizan: quedan con su
    valor inicial como frontera de Dirichlet.

    tol es sobre el residuo RELATIVO ||r|| / ||b||, mas robusto que el
    absoluto cuando la malla es grande.
    Retorna (u, iteraciones, convergio, historial_residuo_relativo).
    """
    nrows, ncols = u.shape
    historial = np.empty(max_iter, dtype=np.float64)
    iteraciones = max_iter
    convergio = False

    norm_b = 0.0
    for i in range(nrows):
        for j in range(ncols):
            norm_b += b[i, j] * b[i, j]
    norm_b = np.sqrt(norm_b)
    if norm_b < 1e-12:
        norm_b = 1e-12

    for k in range(max_iter):
        for i in prange(1, nrows - 1):
            j_inicio = 2 - (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]
                        + b[i, j]) / (4.0 + lam_map[i, j])
                u[i, j] += omega * (u_gs - u[i, j])

        for i in prange(1, nrows - 1):
            j_inicio = 1 + (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1]
                        + b[i, j]) / (4.0 + lam_map[i, j])
                u[i, j] += omega * (u_gs - u[i, j])

        res2 = 0.0
        for i in prange(1, nrows - 1):
            for j in range(1, ncols - 1):
                r = (b[i, j] - (4.0 + lam_map[i, j]) * u[i, j]
                     + u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                res2 += r * r
        historial[k] = np.sqrt(res2) / norm_b

        if historial[k] < tol:
            iteraciones = k + 1
            convergio = True
            break

    return u, iteraciones, convergio, historial[:iteraciones]


def _calentamiento():
    """Fuerza la compilacion JIT de los dos kernels con una malla pequena."""
    u = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros((8, 8), dtype=np.float64)
    m = np.zeros((8, 8), dtype=np.bool_)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    lam = np.zeros((8, 8), dtype=np.float64)
    with _LOCK_KERNEL:
        _kernel_sor_rb(u, b, m, 1.5, 3, 1e-2)
        _kernel_sor_rb_screened(u.copy(), b, lam, 1.5, 3, 1e-2)


def resolver(problema: dict, omega: float = None, tol: float = 1e-6,
             max_iter: int = 2000, ecuacion: str = 'poisson') -> dict:
    """
    Resuelve el problema clasico con mascara (core.poisson_builder.construir_problema).

    ecuacion : 'poisson' (usa b del problema) o 'laplace' (b = 0).
    omega    : None usa problema['omega_opt'].

    Retorna dict con 'imagen_recon', 'iteraciones', 'convergio',
    'historial', 'tiempo', 'omega', 'ecuacion'.
    """
    nrows, ncols = problema['forma']
    if ecuacion == 'poisson':
        b = np.ascontiguousarray(problema['b'], dtype=np.float64)
    else:
        b = np.zeros((nrows, ncols), dtype=np.float64)

    if omega is None:
        omega = problema['omega_opt']

    u = np.ascontiguousarray(problema['imagen_baja'], dtype=np.float64).copy()
    mascara = np.ascontiguousarray(problema['mascara'])

    with _LOCK_KERNEL:
        t0 = time.perf_counter()
        u, iters, convergio, historial = _kernel_sor_rb(
            u, b, mascara, float(omega), int(max_iter), float(tol))
        tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': np.clip(u, 0.0, 1.0),
        'iteraciones':  int(iters),
        'convergio':    bool(convergio),
        'historial':    historial,
        'tiempo':       tiempo,
        'omega':        float(omega),
        'ecuacion':     ecuacion,
    }


def resolver_screened(problema: dict, omega: float = None, tol: float = 1e-6,
                      max_iter: int = 150) -> dict:
    """
    Resuelve un problema "screened" (core.poisson_builder.construir_problema_gradiente).
    No recorta a [0,1]: el modelo RZI recorta al final del pipeline.
    """
    if omega is None:
        omega = problema['omega_opt']

    u = np.ascontiguousarray(problema['imagen_baja'], dtype=np.float64).copy()
    b = np.ascontiguousarray(problema['b'], dtype=np.float64)
    lam_map = np.ascontiguousarray(problema['lam_map'], dtype=np.float64)

    with _LOCK_KERNEL:
        t0 = time.perf_counter()
        u, iters, convergio, historial = _kernel_sor_rb_screened(
            u, b, lam_map, float(omega), int(max_iter), float(tol))
        tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': u,
        'iteraciones':  int(iters),
        'convergio':    bool(convergio),
        'historial':    historial,
        'tiempo':       tiempo,
        'omega':        float(omega),
        'ecuacion':     'screened',
        'lam':          problema.get('lam', 0.0),
    }


def resolver_screened_con_snapshots(problema: dict, omega: float = None,
                                    tol: float = 1e-6, max_iter: int = 150,
                                    cada_n_iter: int = 20) -> dict:
    """
    Resuelve un problema "screened" capturando estados intermedios cada
    'cada_n_iter' iteraciones para la animacion didactica del pipeline.
    """
    if omega is None:
        omega = problema['omega_opt']

    u = np.ascontiguousarray(problema['imagen_baja'], dtype=np.float64).copy()
    b = np.ascontiguousarray(problema['b'], dtype=np.float64)
    lam_map = np.ascontiguousarray(problema['lam_map'], dtype=np.float64)

    snapshots = []
    iteraciones_totales = 0
    convergio = False
    historial_acumulado = []
    paso = max(1, int(cada_n_iter))

    with _LOCK_KERNEL:
        t0 = time.perf_counter()
        for bloque in range(0, int(max_iter), paso):
            iters_bloque = min(paso, int(max_iter) - bloque)
            u, iters, conv, hist = _kernel_sor_rb_screened(
                u, b, lam_map, float(omega), int(iters_bloque), float(tol))
            iteraciones_totales += int(iters)
            historial_acumulado.extend(hist.tolist())
            snapshots.append(u.copy())
            if conv:
                convergio = True
                break
        tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': u,
        'snapshots':    snapshots,
        'iteraciones':  iteraciones_totales,
        'convergio':    convergio,
        'historial':    np.asarray(historial_acumulado, dtype=np.float64),
        'tiempo':       tiempo,
        'omega':        float(omega),
        'ecuacion':     'screened',
        'lam':          problema.get('lam', 0.0),
    }


def barrido_omega(problema: dict, tol: float = 1e-3, max_iter: int = 500,
                  n_puntos: int = 39) -> dict:
    """
    Resuelve el mismo problema con muchos valores de omega en [1.0, 1.99] y
    cuenta las iteraciones. La curva tiene forma de V y su minimo empirico
    se compara con:
      - omega estimado con la mascara (metodo de la potencia + Young), y
      - omega de la formula de malla completa (solo como referencia; no
        aplica cuando hay pixeles fijos en el interior).
    """
    omegas = np.linspace(1.0, 1.99, n_puntos)
    iteraciones = []
    for om in omegas:
        res = resolver(problema, omega=float(om), tol=tol, max_iter=max_iter,
                       ecuacion='poisson')
        iteraciones.append(res['iteraciones'])

    idx = int(np.argmin(iteraciones))
    return {
        'omegas':               omegas.tolist(),
        'iteraciones':          iteraciones,
        'omega_teorico':        problema['omega_opt'],
        'omega_malla_completa': problema.get('omega_malla_completa'),
        'omega_empirico':       float(omegas[idx]),
        'max_iter':             max_iter,
    }


def _sor_python_puro(u, b, mascara, omega, n_iter):
    """Misma logica que _kernel_sor_rb, en Python puro (sin Numba)."""
    nrows, ncols = u.shape
    historial = []
    for _ in range(n_iter):
        for i in range(1, nrows - 1):
            for j in range(2 - (i % 2), ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])
        for i in range(1, nrows - 1):
            for j in range(1 + (i % 2), ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])
        res2 = 0.0
        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                if not mascara[i, j]:
                    r = (b[i, j] - 4.0 * u[i, j]
                         + u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])
                    res2 += r * r
        historial.append(res2 ** 0.5)
    return u, historial


def benchmark_numba_vs_python(problema: dict, n_iter: int = 30) -> dict:
    """
    Compara Numba contra Python puro haciendo EXACTAMENTE el mismo trabajo:
    n_iter iteraciones completas (fase roja, fase negra y residuo), sin
    criterio de parada. Solo se mide el bucle de calculo.
    """
    omega = problema['omega_opt']
    u0 = np.ascontiguousarray(problema['imagen_baja'], dtype=np.float64)
    b = np.ascontiguousarray(problema['b'], dtype=np.float64)
    mascara = np.ascontiguousarray(problema['mascara'])

    _calentamiento()  # que la compilacion JIT no cuente en el tiempo

    u_nb = u0.copy()
    with _LOCK_KERNEL:
        t0 = time.perf_counter()
        u_nb, _, _, _ = _kernel_sor_rb(u_nb, b, mascara, float(omega), int(n_iter), -1.0)
        t_numba = time.perf_counter() - t0

    u_py = u0.copy()
    t1 = time.perf_counter()
    u_py, _ = _sor_python_puro(u_py, b, mascara, float(omega), int(n_iter))
    t_python = time.perf_counter() - t1

    return {
        'tiempo_numba':  t_numba,
        'tiempo_python': t_python,
        'aceleracion':   t_python / t_numba if t_numba > 0 else float('inf'),
        'iteraciones':   int(n_iter),
        'forma':         problema['forma'],
        'numba_activo':  NUMBA_DISPONIBLE,
        'diferencia_max': float(np.max(np.abs(u_nb - u_py))),
    }
