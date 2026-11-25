"""
Capa 3 - Exportacion de resultados y graficas.

  - Guardar imagenes reconstruidas (PNG/JPEG).
  - Exportar la tabla comparativa a CSV.
  - Graficas: convergencia, barrido de omega, comparacion visual y benchmark.
"""

import csv

import numpy as np
from PIL import Image

# El backend de matplotlib lo elige quien importa este modulo: la interfaz
# usa QtAgg y los scripts/pruebas usan Agg (sin ventana).
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


AZUL, ROSA, VERDE, NARANJA, GRIS = '#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9E9E9E'


# Imagenes

def guardar_imagen(imagen: np.ndarray, ruta: str) -> None:
    """Guarda un array float64 [0,1] (gris o RGB) como imagen."""
    img_u8 = np.round(np.clip(imagen, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(img_u8).save(ruta)


# CSV

def exportar_csv(tabla: list, resultado_solver: dict, ruta: str) -> None:
    """
    Exporta la tabla comparativa (PSNR/SSIM/GMSD/nitidez). Las columnas del
    solver solo se llenan en las filas del modelo RZI (clave 'es_rzi').
    """
    campos = ['Modelo', 'PSNR_dB', 'SSIM', 'GMSD', 'Nitidez',
              'Iteraciones_SOR', 'Tiempo_SOR_s', 'Convergio', 'Omega']
    res = resultado_solver or {}
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in tabla:
            es_rzi = bool(fila.get('es_rzi', False))
            writer.writerow({
                'Modelo':          fila.get('modelo', ''),
                'PSNR_dB':         f"{fila.get('psnr', 0):.2f}",
                'SSIM':            f"{fila.get('ssim', 0):.4f}",
                'GMSD':            f"{fila.get('gmsd', 0):.4f}",
                'Nitidez':         f"{fila.get('nitidez', 0):.4f}",
                'Iteraciones_SOR': res.get('iteraciones', '--') if es_rzi else '--',
                'Tiempo_SOR_s':    f"{res.get('tiempo', 0):.4f}" if es_rzi else '--',
                'Convergio':       ('Si' if res.get('convergio') else 'No') if es_rzi else '--',
                'Omega':           f"{res.get('omega', 0):.4f}" if es_rzi else '--',
            })


# Graficas

def figura_a_imagen(fig, dpi: float = 110) -> np.ndarray:
    """
    Dibuja una figura de matplotlib a un array RGB float [0,1].
    No depende del backend activo (usa Agg directamente), asi funciona igual
    en la interfaz, en los scripts y en las pruebas.
    """
    dpi_original = fig.get_dpi()
    fig.set_dpi(dpi)
    try:
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].astype(np.float64) / 255.0
    finally:
        fig.set_dpi(dpi_original)


def _guardar(fig, ruta):
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def grafica_convergencia(historial: np.ndarray, omega: float,
                         error_datos: np.ndarray = None,
                         ruta: str = None, compacta: bool = False) -> plt.Figure:
    """
    Izquierda: residuo relativo del SOR en todas sus iteraciones (cada
    "diente" es un ciclo del modelo, que arranca de una nueva estimacion).
    Derecha (si se da error_datos): error de consistencia ||Down(u) - f||
    despues de cada ciclo, que debe bajar.
    """
    n = 2 if error_datos is not None and len(error_datos) > 1 else 1
    if compacta:   # panel lateral angosto: una grafica debajo de la otra
        fig, ejes = plt.subplots(n, 1, figsize=(5.2, 3.0 * n))
    else:
        fig, ejes = plt.subplots(1, n, figsize=(6 * n, 4.2))
    ejes = np.atleast_1d(ejes)

    ax = ejes[0]
    if len(historial):
        ax.semilogy(np.maximum(historial, 1e-16), color=AZUL, linewidth=1.6,
                    label=f'SOR Red-Black (omega={omega:.3f})')
    ax.set_xlabel('Iteracion SOR acumulada')
    ax.set_ylabel('Residuo relativo (log)')
    ax.set_title('Convergencia del solver')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    if n == 2:
        ax = ejes[1]
        ax.semilogy(np.arange(len(error_datos)), np.maximum(error_datos, 1e-16),
                    'o-', color=VERDE, linewidth=1.6, label='Error ||Down(u) - f||')
        ax.set_xlabel('Ciclo (0 = bicubica inicial)')
        ax.set_ylabel('Error RMS (log)')
        ax.set_title('Consistencia con la imagen original')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle=':', alpha=0.6)

    fig.tight_layout()
    return _guardar(fig, ruta)


def grafica_barrido_omega(omegas: list, iteraciones: list, omega_teorico: float,
                          omega_malla_completa: float = None,
                          ruta: str = None, compacta: bool = False) -> plt.Figure:
    """
    Curva Iteraciones vs Omega (forma de V). Marca el minimo empirico, el
    omega estimado con la mascara (Young + metodo de la potencia) y, como
    referencia, el de la formula de malla completa.
    """
    fig, ax = plt.subplots(figsize=(5.6, 4.0) if compacta else (9, 5))
    ax.plot(omegas, iteraciones, 'o-', color=AZUL, linewidth=2, markersize=4,
            label='Iteraciones medidas')
    idx = int(np.argmin(iteraciones))
    ax.axvline(omegas[idx], color=VERDE, linestyle=':', linewidth=2,
               label=f'minimo empirico = {omegas[idx]:.3f}')
    ax.axvline(omega_teorico, color=ROSA, linestyle='--', linewidth=2,
               label=f'teorico con mascara = {omega_teorico:.3f}')
    if omega_malla_completa is not None:
        ax.axvline(omega_malla_completa, color=GRIS, linestyle='-.', linewidth=1.5,
                   label=f'formula malla completa = {omega_malla_completa:.3f}')
    ax.set_xlabel('Parametro de relajacion omega')
    ax.set_ylabel('Iteraciones hasta converger')
    ax.set_title('Barrido de omega (SOR Red-Black)')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    return _guardar(fig, ruta)


def elegir_recorte(img: np.ndarray, lado: int) -> tuple:
    """Elige la ventana lado x lado con mas bordes (para mostrar detalle)."""
    g = img if img.ndim == 2 else img.mean(axis=2)
    H, W = g.shape
    lado = int(min(lado, H, W))
    gy, gx = np.gradient(g)
    e = np.hypot(gx, gy)
    paso = max(1, lado // 4)
    mejor, pos = -1.0, (0, 0)
    for y in range(0, H - lado + 1, paso):
        for x in range(0, W - lado + 1, paso):
            s = e[y:y + lado, x:x + lado].sum()
            if s > mejor:
                mejor, pos = s, (y, x)
    return pos[0], pos[1], lado


def grafica_comparacion(original: np.ndarray, candidatas: dict, tabla: list = None,
                        recorte: tuple = None, columnas: int = 3,
                        ruta: str = None, factor_eval: int = None,
                        compacta: bool = False) -> plt.Figure:
    """
    Cuadricula de comparacion. El primer panel es la verdad de campo completa
    con un recuadro que marca la zona de detalle; los demas muestran ESE
    detalle ampliado para cada metodo (ahi es donde se nota la nitidez), con
    PSNR y SSIM de la imagen completa en el titulo.
    Sin recorte, cada panel muestra la imagen completa.
    """
    metricas = {f['modelo']: f for f in (tabla or [])}
    h, w = original.shape[:2]
    if recorte is None:
        recorte = (0, 0, min(h, w))
        sin_recorte = True
    else:
        sin_recorte = False
    y, x, lado = recorte

    paneles = [('Original (verdad de campo)', original, True)]
    paneles += [(n, img, False) for n, img in candidatas.items()]
    if not sin_recorte:
        paneles.insert(1, ('Original: detalle', original, False))
    n = len(paneles)
    columnas = min(columnas, n)
    filas = int(np.ceil(n / columnas))
    lado_panel = 2.0 if compacta else 3.6
    fig, ejes = plt.subplots(filas, columnas,
                             figsize=(lado_panel * columnas, (lado_panel + 0.35) * filas),
                             squeeze=False)

    for k, (nombre, img, completa) in enumerate(paneles):
        ax = ejes[k // columnas, k % columnas]
        cmap = 'gray' if img.ndim == 2 else None
        if completa or sin_recorte:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            if not sin_recorte:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), lado, lado, fill=False,
                                           edgecolor=ROSA, linewidth=2))
        else:
            ax.imshow(img[y:y + lado, x:x + lado], cmap=cmap, vmin=0, vmax=1,
                      interpolation='nearest')
        titulo = nombre
        if nombre in metricas:
            m = metricas[nombre]
            titulo += f"\nPSNR {m['psnr']:.2f} dB | SSIM {m['ssim']:.3f}"
        ax.set_title(titulo, fontsize=8 if compacta else 11)
        ax.axis('off')
    for k in range(n, filas * columnas):
        ejes[k // columnas, k % columnas].axis('off')

    if factor_eval is not None:
        fig.suptitle(f'Comparacion con verdad de campo (factor de evaluacion x{factor_eval})',
                     fontsize=11, y=1.01)
    fig.tight_layout()
    return _guardar(fig, ruta)


def grafica_benchmark(resultados: list, ruta: str = None,
                      compacta: bool = False) -> plt.Figure:
    """Barras de tiempo Python puro vs Numba por tamano de malla (log)."""
    tamanios = [f"{r['forma'][0]}x{r['forma'][1]}" for r in resultados]
    t_numba = [r['tiempo_numba'] for r in resultados]
    t_python = [r['tiempo_python'] for r in resultados]
    x = np.arange(len(tamanios))
    fig, ax = plt.subplots(figsize=(5.6, 4.0) if compacta else (8, 5))
    ax.bar(x - 0.2, t_python, 0.38, label='Python puro', color=NARANJA)
    ax.bar(x + 0.2, t_numba, 0.38, label='Numba @njit', color=AZUL)
    for i, r in enumerate(resultados):
        ax.text(x[i], max(t_python[i], t_numba[i]) * 1.15,
                f"x{r['aceleracion']:.0f}", ha='center', fontsize=10)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(tamanios)
    ax.set_ylabel('Tiempo (s, escala log)')
    it = resultados[0]['iteraciones'] if resultados else 0
    ax.set_title(f'Benchmark: mismas {it} iteraciones SOR Red-Black')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    return _guardar(fig, ruta)
