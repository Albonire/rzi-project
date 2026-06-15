"""
Capa 2 - Solver SOR Red-Black con Numba.

Implementa el metodo Successive Over-Relaxation con ordenamiento Rojo-Negro
(seccion 3 del documento de proyecto) utilizando el decorador @njit de Numba
para compilacion a codigo maquina nativo.

Enfoque Matrix-Free (seccion 2.2 y 4):
  La matriz A NUNCA se construye en memoria. Se aplica el estencil de 5 puntos
  directamente sobre la grilla 2D de pixeles. Esto elimina el riesgo de OOM
  (seccion 4: 34 GB para una malla de 256x256 en formato denso).

Ordenamiento Rojo-Negro (seccion 3.2):
  Divide la malla como un tablero de ajedrez en dos conjuntos independientes.
  Fase Roja actualiza simultaneamente todos los pixeles con (i+j) par.
  Fase Negra actualiza simultaneamente todos los pixeles con (i+j) impar.
  Esta separacion rompe la dependencia secuencial del SOR clasico y permite
  que @njit compile el bucle a codigo LLVM nativo.
"""

import numpy as np
import time

try:
    from numba import njit, prange
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    prange = range

    def njit(*args, **kwargs):
        """Decorador no-op cuando Numba no esta instalado."""
        def decorador(f):
            return f
        return decorador


@njit(cache=True, fastmath=True)
def _kernel_sor_rb(u, b, mascara, omega, max_iter, tol):
    """
    Kernel SOR Red-Black compilado con @njit (Numba).

    Parametros de entrada (todos arrays contiguos en memoria):
      u        : float64 (nrows, ncols), estimacion inicial / solucion in-place
      b        : float64 (nrows, ncols), termino independiente (divergencia Poisson)
      mascara  : bool   (nrows, ncols), True donde el pixel es conocido (Dirichlet)
      omega    : float64, factor de sobrerrelajacion en (1, 2)
      max_iter : int, maximo de iteraciones
      tol      : float64, tolerancia de convergencia (norma L2 del residuo)

    Retorna:
      u           : solucion final (modifica en-lugar)
      iteraciones : int, iteraciones realizadas
      convergio   : bool
      historial   : array float64 con el residuo por iteracion
    """
    nrows, ncols = u.shape
    historial    = np.empty(max_iter, dtype=np.float64)
    iteraciones  = max_iter
    convergio    = False

    for k in range(max_iter):

        # ---- Fase Roja: actualizar pixeles donde (i+j) es par ----
        for i in range(1, nrows - 1):
            j_inicio = 2 - (i % 2)   # 2 si fila par, 1 si fila impar
            for j in range(j_inicio, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j]
                            + u[i, j-1] + u[i, j+1]
                            + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])

        # ---- Fase Negra: actualizar pixeles donde (i+j) es impar ----
        for i in range(1, nrows - 1):
            j_inicio = 1 + (i % 2)   # 1 si fila par, 2 si fila impar
            for j in range(j_inicio, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1, j] + u[i+1, j]
                            + u[i, j-1] + u[i, j+1]
                            + b[i, j]) * 0.25
                    u[i, j] += omega * (u_gs - u[i, j])

        # ---- Residuo L2 (criterio de parada) ----
        res2 = 0.0
        for i in range(1, nrows - 1):
            for j in range(1, ncols - 1):
                if not mascara[i, j]:
                    r = (4.0 * u[i, j]
                         - u[i-1, j] - u[i+1, j]
                         - u[i, j-1] - u[i, j+1]
                         - b[i, j])
                    res2 += r * r
        historial[k] = np.sqrt(res2)

        if historial[k] < tol:
            iteraciones = k + 1
            convergio   = True
            break

    return u, iteraciones, convergio, historial[:iteraciones]


@njit(cache=True, fastmath=True, parallel=True)
def _kernel_sor_rb_screened(u, b, lam_map, omega, max_iter, tol):
    """
    Kernel SOR Red-Black para la ecuacion de Poisson "screened" (apantallada):

        (4 + lam_map[i,j]) * u[i,j] - sum(vecinos) = b[i,j]

    Esta es la ecuacion de Euler-Lagrange del funcional
        E(u) = lam * sum_{(i,j) en muestras} (u-f)^2 + sum ||grad(u) - v'||^2
    El termino `lam_map` ancla los pixeles muestreados (fidelidad de datos)
    y `b` incluye -div(v') (campo de gradientes amplificado) mas lam*f en
    los pixeles muestreados. Con lam_map=0 en todas partes se reduce al
    kernel armonico/Poisson original (diagonal 4).

    A diferencia de _kernel_sor_rb, NO recibe mascara: los bordes (fila/
    columna 0 y -1) simplemente no se actualizan (quedan en su valor
    inicial, usado como condicion de frontera tipo Dirichlet con la guia
    bicubica). Las fases Roja y Negra se paralelizan con prange: dentro de
    una fase, cada pixel solo lee vecinos del color opuesto, ya finalizados
    en la fase anterior, por lo que no hay condiciones de carrera.

    Parametros
    ----------
    u        : float64 (nrows, ncols), estimacion inicial / solucion in-place
    b        : float64 (nrows, ncols), termino independiente
    lam_map  : float64 (nrows, ncols), peso de fidelidad (lam en muestras, 0 si no)
    omega    : float64, factor de sobrerrelajacion
    max_iter : int, maximo de iteraciones
    tol      : float64, tolerancia del residuo RELATIVO ||r||/||b||

    Retorna
    -------
    u, iteraciones, convergio, historial (residuo relativo por iteracion)
    """
    nrows, ncols = u.shape
    historial    = np.empty(max_iter, dtype=np.float64)
    iteraciones  = max_iter
    convergio    = False

    norm_b = 0.0
    for i in range(nrows):
        for j in range(ncols):
            norm_b += b[i, j] * b[i, j]
    norm_b = np.sqrt(norm_b)
    if norm_b < 1e-12:
        norm_b = 1e-12

    for k in range(max_iter):

        # ---- Fase Roja: (i+j) par ----
        for i in prange(1, nrows - 1):
            j_inicio = 2 - (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                diag  = 4.0 + lam_map[i, j]
                u_gs  = (u[i-1, j] + u[i+1, j]
                         + u[i, j-1] + u[i, j+1]
                         + b[i, j]) / diag
                u[i, j] += omega * (u_gs - u[i, j])

        # ---- Fase Negra: (i+j) impar ----
        for i in prange(1, nrows - 1):
            j_inicio = 1 + (i % 2)
            for j in range(j_inicio, ncols - 1, 2):
                diag  = 4.0 + lam_map[i, j]
                u_gs  = (u[i-1, j] + u[i+1, j]
                         + u[i, j-1] + u[i, j+1]
                         + b[i, j]) / diag
                u[i, j] += omega * (u_gs - u[i, j])

        # ---- Residuo relativo (criterio de parada) ----
        res2 = 0.0
        for i in prange(1, nrows - 1):
            for j in range(1, ncols - 1):
                diag = 4.0 + lam_map[i, j]
                r = (b[i, j]
                     - diag * u[i, j]
                     + u[i-1, j] + u[i+1, j]
                     + u[i, j-1] + u[i, j+1])
                res2 += r * r
        historial[k] = np.sqrt(res2) / norm_b

        if historial[k] < tol:
            iteraciones = k + 1
            convergio   = True
            break

    return u, iteraciones, convergio, historial[:iteraciones]


def _calentamiento():
    """Fuerza la compilacion JIT de Numba con una malla de juguete."""
    u  = np.zeros((8, 8),  dtype=np.float64)
    b  = np.zeros((8, 8),  dtype=np.float64)
    m  = np.zeros((8, 8),  dtype=np.bool_)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    _kernel_sor_rb(u, b, m, 1.5, 5, 1e-2)

    lam = np.zeros((8, 8), dtype=np.float64)
    _kernel_sor_rb_screened(u.copy(), b, lam, 1.5, 5, 1e-2)


def resolver(problema: dict, omega: float = None, tol: float = 1e-6,
             max_iter: int = 2000, ecuacion: str = 'poisson') -> dict:
    """
    Resuelve la ecuacion de Poisson (o Laplace) usando SOR Red-Black.

    Parametros
    ----------
    problema  : dict devuelto por poisson_builder.construir_problema()
    omega     : factor de relajacion. None -> usa problema['omega_opt']
    tol       : tolerancia de convergencia
    max_iter  : maximo de iteraciones
    ecuacion  : 'poisson' (b = divergencia) o 'laplace' (b = 0)

    Retorna
    -------
    dict con:
      'imagen_recon' : ndarray float64 [0,1], reconstruccion final
      'iteraciones'  : int
      'convergio'    : bool
      'historial'    : ndarray float64, residuo por iteracion
      'tiempo'       : float, segundos de computo
      'omega'        : float, omega efectivamente usado
      'ecuacion'     : str
    """
    nrows, ncols = problema['forma']

    b = (problema['b'].astype(np.float64)
         if ecuacion == 'poisson'
         else np.zeros((nrows, ncols), dtype=np.float64))

    if omega is None:
        omega = problema['omega_opt']

    u       = np.ascontiguousarray(problema['imagen_baja'].copy(), dtype=np.float64)
    b       = np.ascontiguousarray(b)
    mascara = np.ascontiguousarray(problema['mascara'])

    t0 = time.perf_counter()
    u_sol, iters, convergio, historial = _kernel_sor_rb(
        u, b, mascara, float(omega), int(max_iter), float(tol)
    )
    tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': np.clip(u_sol, 0.0, 1.0),
        'iteraciones':  int(iters),
        'convergio':    bool(convergio),
        'historial':    historial,
        'tiempo':       tiempo,
        'omega':        float(omega),
        'ecuacion':     ecuacion,
    }


def resolver_screened(problema: dict, omega: float = None, tol: float = 1e-4,
                       max_iter: int = 600) -> dict:
    """
    Resuelve el problema de Poisson "screened" (apantallado) construido por
    poisson_builder.construir_problema_screened().

    A diferencia de resolver(), no usa 'mascara' ni el argumento 'ecuacion':
    el problema ya trae 'b' (incluye -div(v') + lam*f en las muestras) y
    'lam_map' (peso de fidelidad por pixel). El criterio de parada es el
    residuo RELATIVO ||r||/||b|| < tol (mas robusto que el residuo absoluto
    para mallas grandes, donde el piso de precision de float64 hace que un
    residuo absoluto muy pequeno nunca se alcance).

    Retorna el mismo formato que resolver(), mas 'lam' (valor representativo
    usado para omega_opt).
    """
    nrows, ncols = problema['forma']

    if omega is None:
        omega = problema['omega_opt']

    u       = np.ascontiguousarray(problema['imagen_baja'].copy(), dtype=np.float64)
    b       = np.ascontiguousarray(problema['b'], dtype=np.float64)
    lam_map = np.ascontiguousarray(problema['lam_map'], dtype=np.float64)

    t0 = time.perf_counter()
    u_sol, iters, convergio, historial = _kernel_sor_rb_screened(
        u, b, lam_map, float(omega), int(max_iter), float(tol)
    )
    tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': np.clip(u_sol, 0.0, 1.0),
        'iteraciones':  int(iters),
        'convergio':    bool(convergio),
        'historial':    historial,
        'tiempo':       tiempo,
        'omega':        float(omega),
        'ecuacion':     'screened',
        'lam':          problema.get('lam', 0.0),
    }


def barrido_omega(problema: dict, tol: float = 5e-4, max_iter: int = 300,
                  n_puntos: int = 38) -> dict:
    """
    Barre omega en [1.02, 1.99] y registra el numero de iteraciones.

    Genera los datos para la Prueba de Convergencia Espectral (seccion 8.1):
    la curva tiene forma de V con el minimo exactamente en omega_opt teorico.

    Parametros
    ----------
    problema  : dict de construir_problema()
    tol       : tolerancia relajada para acelerar el barrido
    max_iter  : tope de iteraciones por punto del barrido
    n_puntos  : cantidad de valores de omega evaluados

    Retorna
    -------
    dict con 'omegas', 'iteraciones', 'omega_teorico'
    """
    omegas     = np.linspace(1.02, 1.99, n_puntos)
    iteraciones = []

    for om in omegas:
        res = resolver(problema, omega=float(om), tol=tol,
                       max_iter=max_iter, ecuacion='poisson')
        iteraciones.append(res['iteraciones'])

    return {
        'omegas':       omegas.tolist(),
        'iteraciones':  iteraciones,
        'omega_teorico': problema['omega_opt'],
    }


def benchmark_numba_vs_python(problema: dict, tol: float = 1e-4,
                               max_iter: int = 200) -> dict:
    """
    Compara el tiempo de ejecucion entre Numba y Python puro.
    Necesario para la Prueba de Rendimiento (seccion 8.3).
    """
    omega = problema['omega_opt']

    # --- Numba ---
    t0  = time.perf_counter()
    res = resolver(problema, omega=omega, tol=tol, max_iter=max_iter)
    t_numba = time.perf_counter() - t0
    iters = res['iteraciones']

    # --- Python puro (misma logica, sin @njit) ---
    nrows, ncols = problema['forma']
    u       = problema['imagen_baja'].copy().astype(np.float64)
    b       = problema['b'].astype(np.float64)
    mascara = problema['mascara']

    t1 = time.perf_counter()
    for k in range(max(1, iters)):     # mismas iteraciones que Numba
        for i in range(1, nrows - 1):
            j_ini = 2 - (i % 2)
            for j in range(j_ini, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]+b[i,j])*0.25
                    u[i, j] += omega * (u_gs - u[i, j])
        for i in range(1, nrows - 1):
            j_ini = 1 + (i % 2)
            for j in range(j_ini, ncols - 1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]+b[i,j])*0.25
                    u[i, j] += omega * (u_gs - u[i, j])
    t_python = time.perf_counter() - t1

    aceleracion = t_python / t_numba if t_numba > 0 else float('inf')

    return {
        'tiempo_numba':   t_numba,
        'tiempo_python':  t_python,
        'aceleracion':    aceleracion,
        'iteraciones':    iters,
        'forma':          problema['forma'],
        'numba_activo':   NUMBA_DISPONIBLE,
    }


def resolver_con_snapshots(problema: dict, omega: float = None, tol: float = 1e-6,
                           max_iter: int = 2000, ecuacion: str = 'poisson',
                           cada_n_iter: int = 20) -> dict:
    """
    Resuelve la ecuacion capturando snapshots del estado de la malla
    cada N iteraciones. Util para animaciones de convergencia.

    Retorna un dict con los mismos campos que resolver() mas:
    'snapshots': lista de ndarrays float64 [0,1] con el estado cada N iteraciones.
    """
    import numpy as np
    import time

    nrows, ncols = problema['forma']
    b = (problema['b'].astype(np.float64)
         if ecuacion == 'poisson'
         else np.zeros((nrows, ncols), dtype=np.float64))

    if omega is None:
        omega = problema['omega_opt']

    u       = np.ascontiguousarray(problema['imagen_baja'].copy(), dtype=np.float64)
    b       = np.ascontiguousarray(b)
    mascara = np.ascontiguousarray(problema['mascara'])

    snapshots = []
    t0 = time.perf_counter()

    snapshots.append(np.clip(u.copy(), 0.0, 1.0))

    iteraciones_totales = 0
    convergio = False
    historial_acumulado = []

    for bloque in range(0, max_iter, cada_n_iter):
        iters_bloque = min(cada_n_iter, max_iter - bloque)
        u_sol, iters, conv, hist = _kernel_sor_rb(
            u, b, mascara, float(omega), int(iters_bloque), float(tol)
        )
        iteraciones_totales += iters
        historial_acumulado.extend(hist.tolist())

        snapshots.append(np.clip(u_sol.copy(), 0.0, 1.0))

        if conv:
            convergio = True
            break

    tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': np.clip(u, 0.0, 1.0),
        'snapshots': snapshots,
        'iteraciones': iteraciones_totales,
        'convergio': convergio,
        'historial': np.array(historial_acumulado),
        'tiempo': tiempo,
        'omega': float(omega),
        'ecuacion': ecuacion,
        'cada_n_iter': cada_n_iter,
    }


def resolver_screened_con_snapshots(problema: dict, omega: float = None, tol: float = 1e-4,
                                     max_iter: int = 600, cada_n_iter: int = 8) -> dict:
    """
    Version de resolver_screened() que captura snapshots del estado de la
    malla cada 'cada_n_iter' iteraciones (incluye el estado inicial, la guia
    bicubica). Util para animar la convergencia del modelo screened.

    Retorna el mismo formato que resolver_screened(), mas:
    'snapshots'    : lista de ndarrays float64 [0,1] (cada uno en la grilla
                     de esta etapa).
    'cada_n_iter'  : int.
    """
    nrows, ncols = problema['forma']

    if omega is None:
        omega = problema['omega_opt']

    u       = np.ascontiguousarray(problema['imagen_baja'].copy(), dtype=np.float64)
    b       = np.ascontiguousarray(problema['b'], dtype=np.float64)
    lam_map = np.ascontiguousarray(problema['lam_map'], dtype=np.float64)

    snapshots = []
    t0 = time.perf_counter()

    snapshots.append(np.clip(u.copy(), 0.0, 1.0))

    iteraciones_totales = 0
    convergio = False
    historial_acumulado = []

    for bloque in range(0, max_iter, cada_n_iter):
        iters_bloque = min(cada_n_iter, max_iter - bloque)
        u, iters, conv, hist = _kernel_sor_rb_screened(
            u, b, lam_map, float(omega), int(iters_bloque), float(tol)
        )
        iteraciones_totales += iters
        historial_acumulado.extend(hist.tolist())

        snapshots.append(np.clip(u.copy(), 0.0, 1.0))

        if conv:
            convergio = True
            break

    tiempo = time.perf_counter() - t0

    return {
        'imagen_recon': np.clip(u, 0.0, 1.0),
        'snapshots': snapshots,
        'iteraciones': iteraciones_totales,
        'convergio': convergio,
        'historial': np.array(historial_acumulado),
        'tiempo': tiempo,
        'omega': float(omega),
        'ecuacion': 'screened',
        'lam': problema.get('lam', 0.0),
        'cada_n_iter': cada_n_iter,
    }
