"""
Capa 2 - Constructor del Problema de Poisson.

Implementa el Enfoque B (Verdad de Campo) descrito en la seccion 1.2
del documento de proyecto:
  1. Se recibe una imagen de alta resolucion (verdad de campo).
  2. Se submuestrea: solo 1 de cada 'factor^2' pixeles es considerado conocido.
  3. Se calcula el campo de gradientes de la imagen de baja resolucion.
  4. Se calcula la divergencia discreta del campo (termino b de Poisson).

La ecuacion que se resuelve para cada pixel desconocido (i, j) es:
    4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = b[i,j]

donde b[i,j] = divergencia del campo de gradientes (NO es cero como en Laplace).
Para pixeles conocidos: u[i,j] = valor_conocido (condicion de Dirichlet).
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def imagen_a_grises(imagen: np.ndarray) -> np.ndarray:
    """
    Convierte cualquier imagen a escala de grises float64 en [0, 1].
    Usa la formula de luminancia ITU-R BT.601.
    """
    img = np.asarray(imagen, dtype=np.float64)
    if img.max() > 1.01:
        img = img / 255.0
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] >= 3:
            return (0.2989 * img[:, :, 0]
                    + 0.5870 * img[:, :, 1]
                    + 0.1140 * img[:, :, 2])
        return img[:, :, 0]
    raise ValueError(f"Formato de imagen no soportado: {imagen.shape}")


def calcular_omega_optimo(nrows: int, ncols: int) -> float:
    """
    Parametro de relajacion optimo para SOR en malla nrows x ncols.

    Formula (Young, 1971; seccion 3.3 del documento):
        omega_opt = 2 / (1 + sin(pi / (n + 1)))
    donde n = min(nrows, ncols).

    Este valor es el acelerador espectral que minimiza el radio de convergencia
    de la matriz de iteracion SOR. No es un filtro de imagen.
    """
    n = min(nrows, ncols)
    return 2.0 / (1.0 + np.sin(np.pi / (n + 1)))


def construir_problema(imagen_gris: np.ndarray, factor: int = 4) -> dict:
    """
    Prepara el problema de Poisson a partir de una imagen de alta resolucion.

    Parametros
    ----------
    imagen_gris : ndarray float64 [0,1], shape (nrows, ncols)
        Imagen de alta resolucion que sirve como verdad de campo.
    factor : int
        Factor de submuestreo. Con factor=4, solo 1 de cada 16 pixeles
        es considerado "conocido". Los demas deben ser reconstruidos.

    Retorna
    -------
    dict con las claves:
        'imagen_original' : ndarray float64, la verdad de campo.
        'imagen_baja'     : ndarray float64, solo pixeles conocidos (resto en 0).
        'mascara'         : ndarray bool, True donde el pixel es conocido.
        'b'               : ndarray float64, divergencia del campo de gradientes.
        'forma'           : tuple (nrows, ncols).
        'omega_opt'       : float, parametro omega optimo para SOR.
        'factor'          : int, factor de submuestreo utilizado.
    """
    if imagen_gris.ndim != 2:
        raise ValueError("Se espera imagen 2D en escala de grises.")

    nrows, ncols = imagen_gris.shape

    # --- Mascara de pixeles conocidos (submuestreo uniforme + bordes) ---
    mascara = np.zeros((nrows, ncols), dtype=np.bool_)
    mascara[::factor, ::factor] = True
    mascara[0, :]  = True
    mascara[-1, :] = True
    mascara[:, 0]  = True
    mascara[:, -1] = True

    # --- Imagen de baja resolucion (pixeles desconocidos valen cero) ---
    imagen_baja = np.zeros((nrows, ncols), dtype=np.float64)
    imagen_baja[mascara] = imagen_gris[mascara]

    # --- Campo guia para la divergencia (Poisson b vector) ---
    # Rellenar pixeles desconocidos con el vecino conocido mas cercano antes
    # de calcular gradientes. Esto evita los artefactos que genera operar
    # sobre los ceros de imagen_baja (mitigacion del riesgo, seccion 10).
    from scipy.ndimage import distance_transform_edt
    fill_idx = distance_transform_edt(~mascara,
                                       return_distances=False,
                                       return_indices=True)
    guia = imagen_gris[fill_idx[0], fill_idx[1]]

    # Suavizado Gaussiano para reducir artefactos de cuantizacion
    suavizada = gaussian_filter(guia, sigma=1.0)

    gy = np.gradient(suavizada, axis=0)  # gradiente vertical
    gx = np.gradient(suavizada, axis=1)  # gradiente horizontal

    # --- Divergencia discreta: b[i,j] = d(gx)/dx + d(gy)/dy ---
    b = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)

    # En pixeles conocidos la condicion de Dirichlet domina; b no interviene.
    b[mascara] = 0.0

    return {
        'imagen_original': imagen_gris.astype(np.float64),
        'imagen_baja':     imagen_baja,
        'mascara':         mascara,
        'b':               b.astype(np.float64),
        'forma':           (nrows, ncols),
        'omega_opt':       calcular_omega_optimo(nrows, ncols),
        'factor':          factor,
    }


def construir_problema_upscale(tile: np.ndarray, factor: int = 4) -> dict:
    """
    construye un problema de upscaling real: agranda el tile factor veces
    usando la pde para rellenar los pixeles intermedios.

    a diferencia de construir_problema (que degrada la imagen para medir calidad),
    aqui el tile ES la imagen de entrada. se crea una malla fina donde los pixeles
    del tile son puntos conocidos exactos y la pde rellena los huecos.

    la malla queda alineada por esquinas:
      - tile[i, j] cae exactamente en malla[i*factor, j*factor]
      - no quedan filas ni columnas colgando en los bordes

    se usa la interpolacion bicubica como guia (punto de partida del solver y
    termino b). esto logra convergencia rapida y elimina los artefactos de puntitos.

    Parametros
    ----------
    tile : ndarray float64 [0,1], shape (h, w)
        Region de la imagen que se desea agrandar.
    factor : int
        Factor de ampliacion. factor=4 produce una imagen ~4x mas grande por lado.

    Retorna
    -------
    dict con las mismas claves que construir_problema (reutiliza resolver() sin cambios).
        'imagen_original' : guia bicubica como referencia
        'imagen_baja'     : guia bicubica con valores exactos del tile
        'mascara'         : True en los pixeles originales del tile
        'b'               : laplaciano discreto de la guia (5 puntos)
        'forma'           : (new_h, new_w) de la malla ampliada
        'omega_opt'       : omega optimo para la malla nueva
        'factor'          : factor de ampliacion
    """
    if tile.ndim != 2:
        raise ValueError("Se espera tile 2D en escala de grises.")

    h, w = tile.shape

    # tope de memoria/rendimiento: reducir el tile si la salida seria muy grande
    MAX_PIXELS = 1_200_000
    MAX_LADO   = 1200
    new_h_pre  = (h - 1) * factor + 1
    new_w_pre  = (w - 1) * factor + 1

    if new_h_pre * new_w_pre > MAX_PIXELS or max(new_h_pre, new_w_pre) > MAX_LADO:
        escala = min(
            (MAX_PIXELS / (new_h_pre * new_w_pre)) ** 0.5,
            MAX_LADO / max(new_h_pre, new_w_pre),
        )
        h_small = max(2, (int(new_h_pre * escala) - 1) // factor + 1)
        w_small = max(2, (int(new_w_pre * escala) - 1) // factor + 1)
        pil_small = Image.fromarray(
            (np.clip(tile, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
        pil_small = pil_small.resize((w_small, h_small), Image.LANCZOS)
        tile = np.asarray(pil_small, dtype=np.float64) / 255.0
        h, w = tile.shape

    # malla fina alineada por esquinas
    new_h = (h - 1) * factor + 1
    new_w = (w - 1) * factor + 1

    # guia bicubica: upsample del tile con PIL BICUBIC
    pil_tile = Image.fromarray(
        (np.clip(tile, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
    pil_guia = pil_tile.resize((new_w, new_h), Image.BICUBIC)
    G = np.asarray(pil_guia, dtype=np.float64) / 255.0

    # mascara: los pixeles del tile original son puntos conocidos exactos
    mascara = np.zeros((new_h, new_w), dtype=np.bool_)
    mascara[::factor, ::factor] = True

    # imagen de partida: bicubica con valores exactos en los puntos conocidos
    # la bicubica acelera la convergencia (ya es una buena solucion inicial)
    imagen_baja = G.copy()
    imagen_baja[::factor, ::factor] = tile

    # b = 0 (laplace/armonica): interpolacion mas nitida que bicubica
    # usar b=Lap(G) sesga la solucion hacia la bicubica (borrosa)
    # con b=0 el solver converge a la funcion armonica exacta entre los
    # puntos conocidos: preserva bordes y no tiene ringing de la bicubica
    b = np.zeros((new_h, new_w), dtype=np.float64)

    return {
        'imagen_original': G,
        'imagen_baja':     imagen_baja,
        'mascara':         mascara,
        'b':               b,
        'forma':           (new_h, new_w),
        'omega_opt':       calcular_omega_optimo(new_h, new_w),
        'factor':          factor,
    }
