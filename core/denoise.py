"""
Capa 2 - Limpieza de la imagen de entrada antes de ampliar.

Por que hace falta: el modelo RZI obliga, con la back-projection, a que la
imagen ampliada vuelva a dar EXACTAMENTE la imagen pequena al reducirla. Si
la imagen pequena trae ruido o bloques de JPEG, el modelo los reproduce con
fidelidad (y los refuerza en los bordes). Medido con las imagenes de
ejemplo del equipo: con ruido de 0.02 RZI quedaba 1.9 dB por debajo de la
bicubica; limpiando antes la entrada queda 2.5 dB por encima.

Dos piezas:

1. estimar_ruido(): desviacion estandar del ruido, medida solo en las zonas
   planas de la imagen (Immerkaer, 1996, con exclusion de bordes como en
   Tai y Yang, 2008). En zonas planas lo que queda tras un filtro de
   "segunda diferencia" es casi solo ruido; en los bordes y texturas no, por
   eso se excluyen (si no, la textura se confundiria con ruido).

2. limpiar(): variacion total (modelo ROF, Rudin, Osher y Fatemi, 1992):
       min_u  sum |grad u|  +  (1 / (2*peso)) * sum (u - f)^2
   Es una EDP (su ecuacion de Euler-Lagrange es una difusion no lineal) que
   aplana el ruido pero conserva los bordes, a diferencia de un desenfoque.
   Se resuelve con el algoritmo de Chambolle (scikit-image).
"""

import numpy as np
from scipy.ndimage import convolve, sobel
from skimage.restoration import denoise_tv_chambolle

# Nucleo de Immerkaer: combinacion de segundas diferencias que anula las
# variaciones suaves (planos y rampas) y deja pasar el ruido.
_NUCLEO = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)

# Ajustados con scripts/evaluar_imagenes.py sobre las imagenes de ejemplo.
UMBRAL_AUTO = 0.013      # sigma minimo para limpiar en modo automatico (ver docstring)
FACTOR_RUIDO = 0.7       # peso de la variacion total = FACTOR_RUIDO * sigma
PESO_JPEG = 0.01         # peso fijo para suavizar bloques de JPEG

MODOS = ('auto', 'ruido', 'jpeg', 'no')

# Con menos pixeles el estimador confunde textura fina (mallas, pelo) con
# ruido. En modo automatico, una region mas chica no se limpia salvo que se
# entregue un sigma medido sobre una zona mas grande (la imagen completa).
MIN_PIXELES_AUTO = 128 * 128


def estimar_ruido(img: np.ndarray, fraccion: float = 0.1) -> float:
    """Sigma del ruido (en [0,1]) medido en el 10 % de pixeles mas planos."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    if min(img.shape) < 8:
        return 0.0
    gradiente = np.hypot(sobel(img, 0), sobel(img, 1))[1:-1, 1:-1]
    respuesta = np.abs(convolve(img, _NUCLEO))[1:-1, 1:-1]
    planos = gradiente <= np.quantile(gradiente, fraccion)
    return float(np.sqrt(np.pi / 2) / 6 * respuesta[planos].mean())


def decidir(modo: str, sigma: float, confiable: bool = True) -> str:
    """Que limpieza aplicar: 'ruido', 'jpeg' o 'no'."""
    if modo not in MODOS:
        raise ValueError(f"limpieza debe ser uno de {MODOS}, no {modo!r}")
    if modo == 'auto':
        return 'ruido' if (confiable and sigma > UMBRAL_AUTO) else 'no'
    return modo


def limpiar(img: np.ndarray, modo: str, sigma: float) -> np.ndarray:
    """Aplica la variacion total segun el modo ('ruido', 'jpeg' o 'no')."""
    if modo == 'no':
        return img
    peso = FACTOR_RUIDO * max(sigma, 1e-4) if modo == 'ruido' else PESO_JPEG
    return np.clip(denoise_tv_chambolle(np.asarray(img, dtype=np.float64), weight=peso),
                   0.0, 1.0)


def preparar_entrada(img: np.ndarray, modo: str = 'auto', sigma: float = None) -> tuple:
    """
    Estima el ruido y limpia si corresponde. Retorna
    (imagen_limpia, sigma_estimado, limpieza_aplicada).

    sigma: ruido ya medido sobre una zona mas grande (por ejemplo la imagen
    completa). Si no se da, se mide sobre 'img'; en modo automatico esa
    medida solo se usa si la region tiene al menos MIN_PIXELES_AUTO pixeles.
    """
    if sigma is None:
        sigma = estimar_ruido(img)
        confiable = np.asarray(img).shape[0] * np.asarray(img).shape[1] >= MIN_PIXELES_AUTO
    else:
        confiable = True
    aplicada = decidir(modo, sigma, confiable)
    return limpiar(img, aplicada, sigma), sigma, aplicada
