"""
Capa 3 - Exportacion de resultados y graficas.

Provee funciones para:
  - Guardar imagenes reconstruidas en disco.
  - Exportar la tabla comparativa a CSV.
  - Generar y guardar las graficas requeridas por el plan de pruebas
    (secciones 8.1, 8.2 y 8.3 del documento de proyecto).
"""

import csv
import os
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')   # backend sin ventana (para guardar graficas)
import matplotlib.pyplot as plt


# Guardar imagenes

def guardar_imagen(imagen: np.ndarray, ruta: str) -> None:
    """Guarda un array float64 [0,1] como imagen PNG o JPEG en disco."""
    img = np.clip(imagen, 0.0, 1.0)
    img_uint8 = (img * 255).astype(np.uint8)
    modo = 'L' if img_uint8.ndim == 2 else 'RGB'
    Image.fromarray(img_uint8, mode=modo).save(ruta)


# Exportar CSV

def exportar_csv(tabla: list, resultado_solver: dict, ruta: str) -> None:
    """
    Exporta la tabla comparativa (PSNR/SSIM) y los parametros del solver.

    Parametros
    ----------
    tabla           : lista de dicts de services.metrics.tabla_comparativa()
    resultado_solver: dict de core.sor_solver.resolver()
    ruta            : ruta del archivo CSV de salida
    """
    campos = ['Modelo', 'PSNR_dB', 'SSIM', 'Iteraciones',
              'Tiempo_s', 'Convergio', 'Omega', 'Ecuacion']
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for fila in tabla:
            es_poisson = 'Poisson' in fila.get('modelo', '')
            writer.writerow({
                'Modelo':      fila.get('modelo', ''),
                'PSNR_dB':     f"{fila.get('psnr', 0):.2f}",
                'SSIM':        f"{fila.get('ssim', 0):.4f}",
                'Iteraciones': resultado_solver.get('iteraciones', '--') if es_poisson else '--',
                'Tiempo_s':    f"{resultado_solver.get('tiempo', 0):.4f}" if es_poisson else '--',
                'Convergio':   ('Si' if resultado_solver.get('convergio', False)
                                else 'No') if es_poisson else '--',
                'Omega':       f"{resultado_solver.get('omega', 0):.4f}" if es_poisson else '--',
                'Ecuacion':    resultado_solver.get('ecuacion', '--') if es_poisson else '--',
            })


# Graficas

def grafica_convergencia(historial: np.ndarray, omega: float,
                          ruta: str = None) -> plt.Figure:
    """
    Genera la grafica de Residuo vs Iteraciones (escala logaritmica).
    Prueba de convergencia del solver SOR Red-Black.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(historial, color='#2196F3', linewidth=2, label=f'SOR Red-Black (omega={omega:.4f})')
    ax.set_xlabel('Iteracion', fontsize=12)
    ax.set_ylabel('Residuo L2 (escala log)', fontsize=12)
    ax.set_title('Convergencia del Solver SOR Red-Black', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def grafica_barrido_omega(omegas: list, iteraciones: list,
                           omega_teorico: float, ruta: str = None) -> plt.Figure:
    """
    Genera la Prueba de Convergencia Espectral (seccion 8.1).

    Muestra la curva Iteraciones vs Omega con forma de V.
    El minimo de la curva debe coincidir con omega_opt teorico.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(omegas, iteraciones, 'o-', color='#2196F3', linewidth=2,
            markersize=4, label='Iteraciones empiricas')
    ax.axvline(omega_teorico, color='#E91E63', linestyle='--', linewidth=2,
               label=f'omega_opt teorico = {omega_teorico:.4f}')

    # Marcar el minimo empirico
    idx_min = int(np.argmin(iteraciones))
    ax.axvline(omegas[idx_min], color='#4CAF50', linestyle=':', linewidth=1.5,
               label=f'omega_opt empirico = {omegas[idx_min]:.4f}')

    ax.set_xlabel('Parametro de relajacion omega', fontsize=12)
    ax.set_ylabel('Numero de iteraciones', fontsize=12)
    ax.set_title('Convergencia Espectral: Iteraciones vs Omega (SOR Red-Black)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def grafica_comparacion_4modelos(original, bicubica, laplace, poisson,
                                  tabla: list, ruta: str = None) -> plt.Figure:
    """
    Genera la figura comparativa de 4 modelos (seccion 8.2):
      Original (Verdad de Campo) | Bicubica | Laplace Puro | Poisson SOR
    Incluye PSNR y SSIM como subtitulo de cada panel.
    """
    fig, ejes = plt.subplots(1, 4, figsize=(16, 4))
    imagenes  = [original, bicubica, laplace, poisson]
    titulos   = ['Original\n(Verdad de Campo)', 'Bicubica\n(Pillow BICUBIC)',
                 'Laplace Puro\n(modelo antiguo)', 'Poisson SOR\n(nuestro modelo)']

    for ax, img, titulo, fila in zip(ejes, imagenes, titulos,
                                     [None] + tabla):
        ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(titulo, fontsize=10, fontweight='bold')
        ax.axis('off')
        if fila:
            ax.set_xlabel(f"PSNR={fila['psnr']:.1f} dB  SSIM={fila['ssim']:.3f}",
                          fontsize=8)

    fig.suptitle('Comparacion Visual de Metodos de Reconstruccion', fontsize=12)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def grafica_benchmark(resultados: list, ruta: str = None) -> plt.Figure:
    """
    Genera la grafica de benchmarking Numba vs Python (seccion 8.3).
    resultados: lista de dicts de sor_solver.benchmark_numba_vs_python()
    """
    tamanios = [f"{r['forma'][0]}x{r['forma'][1]}" for r in resultados]
    t_numba  = [r['tiempo_numba']  for r in resultados]
    t_python = [r['tiempo_python'] for r in resultados]

    x   = np.arange(len(tamanios))
    fig, ax = plt.subplots(figsize=(8, 5))
    bar1 = ax.bar(x - 0.2, t_python, 0.35, label='Python puro', color='#FF9800')
    bar2 = ax.bar(x + 0.2, t_numba,  0.35, label='Numba @njit', color='#2196F3')

    ax.set_xticks(x)
    ax.set_xticklabels(tamanios, fontsize=11)
    ax.set_ylabel('Tiempo de ejecucion (s)', fontsize=12)
    ax.set_title('Benchmarking: SOR Secuencial Python vs SOR Red-Black Numba',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    if ruta:
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return fig


def guardar_grafica(figura: plt.Figure, ruta: str) -> None:
    """Guarda cualquier figura matplotlib en disco."""
    figura.savefig(ruta, dpi=150, bbox_inches='tight')
