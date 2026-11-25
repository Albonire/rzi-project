"""
Genera las tablas y figuras de validacion del proyecto RZI.

Uso (desde la raiz del proyecto):
    python scripts/generar_resultados.py

Salidas:
    docs/resultados/*.csv   tablas de metricas
    docs/figuras/*.png      figuras usadas en el documento de estudio

Imagenes de prueba: las de scikit-image (camera, astronaut, coffee, chelsea),
de uso libre y estandar en procesamiento de imagenes.
"""

import csv
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage import data

from core.image_ops import resize_float, resize_rgb_float
from core.poisson_builder import (construir_problema, imagen_a_grises,
                                  amplified_guidance)
from core.sor_solver import barrido_omega, benchmark_numba_vs_python, _calentamiento
from core.upscaler import upscale_rzi, upscale, PARAMS_DEFECTO
from services.metrics import (degradar, interpolacion_bicubica, interpolacion_lanczos,
                              psnr, ssim, gmsd, nitidez_gradiente)
from services.exporter import (grafica_barrido_omega, grafica_convergencia,
                               grafica_benchmark)

FIG = os.path.join(RAIZ, 'docs', 'figuras')
RES = os.path.join(RAIZ, 'docs', 'resultados')
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

AZUL, ROSA, VERDE, NARANJA, ROJO, GRIS = ('#1976D2', '#D81B60', '#2E7D32',
                                         '#EF6C00', '#C62828', '#757575')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'figure.dpi': 110})


def gris(a):
    return imagen_a_grises(np.asarray(a, dtype=np.float64) / 255.0)


IMAGENES = {
    'camera':    data.camera().astype(np.float64) / 255.0,
    'astronaut': gris(data.astronaut()),
    'coffee':    gris(data.coffee()),
    'chelsea':   gris(data.chelsea()),
}


def vecino(lr, f):
    h, w = lr.shape
    return np.clip(resize_float(lr, w * f, h * f, Image.NEAREST), 0, 1)


def guardar(fig, nombre):
    fig.savefig(os.path.join(FIG, nombre), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  figura', nombre)


# 1. Tabla principal de metricas

def tabla_principal():
    print('Tabla principal (degradacion bicubica)...')
    filas = []
    for nombre, img in IMAGENES.items():
        for f in (2, 3, 4):
            hr, lr = degradar(img, f)
            t0 = time.perf_counter()
            res = upscale_rzi(lr, f)
            t_rzi = time.perf_counter() - t0
            candidatas = {
                'Vecino': vecino(lr, f),
                'Bicubica': interpolacion_bicubica(lr, f),
                'Lanczos': interpolacion_lanczos(lr, f),
                'RZI': res['imagen_recon'],
                'RZI+realce': upscale_rzi(lr, f, realce=0.6)['imagen_recon'],
            }
            for metodo, u in candidatas.items():
                filas.append({
                    'imagen': nombre, 'factor': f, 'metodo': metodo,
                    'psnr': psnr(hr, u), 'ssim': ssim(hr, u), 'gmsd': gmsd(hr, u),
                    'nitidez': nitidez_gradiente(u),
                    'nitidez_hr': nitidez_gradiente(hr),
                    'tiempo_s': t_rzi if metodo == 'RZI' else '',
                    'iter_sor': res['iteraciones'] if metodo == 'RZI' else '',
                })
    with open(os.path.join(RES, 'metricas_principales.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for fi in filas:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in fi.items()})
    return filas


def resumen_ganancias(filas):
    """Promedios por factor y figura de ganancias sobre la bicubica."""
    print('Resumen de ganancias...')
    idx = {(f['imagen'], f['factor'], f['metodo']): f for f in filas}
    resumen = []
    for f in (2, 3, 4):
        for metodo in ('Vecino', 'Bicubica', 'Lanczos', 'RZI', 'RZI+realce'):
            sel = [idx[(n, f, metodo)] for n in IMAGENES]
            resumen.append({
                'factor': f, 'metodo': metodo,
                'psnr': np.mean([s['psnr'] for s in sel]),
                'ssim': np.mean([s['ssim'] for s in sel]),
                'gmsd': np.mean([s['gmsd'] for s in sel]),
                'nitidez_rel': np.mean([s['nitidez'] / s['nitidez_hr'] for s in sel]),
            })
    with open(os.path.join(RES, 'resumen_por_factor.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(resumen[0].keys()))
        w.writeheader()
        for r in resumen:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in r.items()})

    # figura: ganancia de PSNR sobre la bicubica por imagen y factor
    fig, ax = plt.subplots(figsize=(9, 4.5))
    nombres = list(IMAGENES)
    ancho = 0.25
    x = np.arange(len(nombres))
    for k, f in enumerate((2, 3, 4)):
        g = [idx[(n, f, 'RZI')]['psnr'] - idx[(n, f, 'Bicubica')]['psnr'] for n in nombres]
        barras = ax.bar(x + (k - 1) * ancho, g, ancho, label=f'x{f}',
                        color=(AZUL, VERDE, NARANJA)[k])
        for b, v in zip(barras, g):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f'{v:+.2f}', ha='center',
                    fontsize=8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylabel('Ganancia de PSNR sobre bicubica (dB)')
    ax.set_title('RZI frente a la bicubica: cuanto mejora cada imagen')
    ax.legend(title='Factor')
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    guardar(fig, 'ganancia_psnr.png')
    return resumen


# 2. Robustez ante otras degradaciones

def robustez():
    print('Robustez ante otras degradaciones...')

    def deg_lanczos(hr, f):
        H, W = hr.shape
        return np.clip(resize_float(hr, W // f, H // f, Image.LANCZOS), 0, 1)

    def deg_gauss(hr, f):
        return gaussian_filter(hr, 0.5 * f)[f // 2::f, f // 2::f]

    def deg_box(hr, f):
        H, W = hr.shape
        return np.clip(resize_float(hr, W // f, H // f, Image.BOX), 0, 1)

    filas = []
    for etiqueta, deg in (('Bicubica (supuesta por el modelo)', None),
                          ('Lanczos', deg_lanczos), ('Desenfoque gaussiano', deg_gauss),
                          ('Promedio de area (box)', deg_box)):
        g_bic, g_lan = [], []
        for img in IMAGENES.values():
            for f in (2, 4):
                hr, lr = degradar(img, f)
                if deg is not None:
                    lr = deg(hr, f)
                bic = interpolacion_bicubica(lr, f)
                u = upscale_rzi(lr, f)['imagen_recon']
                g_bic.append(psnr(hr, u) - psnr(hr, bic))
                g_lan.append(psnr(hr, u) - psnr(hr, interpolacion_lanczos(lr, f)))
        filas.append({'degradacion': etiqueta, 'ganancia_media_vs_bicubica': np.mean(g_bic),
                      'ganancia_min_vs_bicubica': np.min(g_bic),
                      'ganancia_media_vs_lanczos': np.mean(g_lan)})
    with open(os.path.join(RES, 'robustez.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for r in filas:
            w.writerow({k: (f'{v:.3f}' if isinstance(v, float) else v) for k, v in r.items()})
    return filas


# 3. Figuras visuales

def comparacion_visual():
    print('Figuras de comparacion visual...')
    img = IMAGENES['camera']
    f = 4
    hr, lr = degradar(img, f)
    res = upscale_rzi(lr, f)
    paneles = [('Original (verdad)', hr), ('Vecino mas proximo', vecino(lr, f)),
               ('Bicubica', interpolacion_bicubica(lr, f)),
               ('Lanczos', interpolacion_lanczos(lr, f)),
               ('RZI (este proyecto)', res['imagen_recon']),
               ('RZI + realce 0.6', upscale_rzi(lr, f, realce=0.6)['imagen_recon'])]
    y, x, lado = 60, 200, 150
    fig, ejes = plt.subplots(2, 3, figsize=(12, 8.4))
    for ax, (t, u) in zip(ejes.ravel(), paneles):
        ax.imshow(u[y:y + lado, x:x + lado], cmap='gray', vmin=0, vmax=1,
                  interpolation='nearest')
        sub = '' if t.startswith('Original') else f'\nPSNR {psnr(hr, u):.2f} dB  SSIM {ssim(hr, u):.3f}'
        ax.set_title(t + sub)
        ax.axis('off')
    fig.suptitle('Ampliacion x4 de "camera" (detalle). La imagen pequena tenia 128x128 px.',
                 fontsize=12)
    fig.tight_layout()
    guardar(fig, 'comparacion_camera_x4.png')

    # mapa de error
    bic = interpolacion_bicubica(lr, f)
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, (t, u) in zip(ejes, (('Error de la bicubica', bic),
                                 ('Error de RZI', res['imagen_recon']))):
        im = ax.imshow(np.abs(hr - u)[y:y + lado, x:x + lado], cmap='inferno', vmin=0, vmax=0.25)
        ax.set_title(f'{t}  (error medio {np.mean(np.abs(hr - u)):.4f})')
        ax.axis('off')
    fig.colorbar(im, ax=ejes, fraction=0.025, label='|original - ampliada|')
    guardar(fig, 'mapa_error.png')

    # perfil a traves de un borde
    fila = y + 75
    cols = slice(x, x + lado)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(hr[fila, cols], color='black', linewidth=2.2, label='Original (verdad)')
    ax.plot(bic[fila, cols], color=NARANJA, linewidth=1.8, label='Bicubica')
    ax.plot(res['imagen_recon'][fila, cols], color=VERDE, linewidth=1.8, label='RZI')
    ax.set_xlabel('Posicion a lo largo de una fila (pixeles)')
    ax.set_ylabel('Intensidad (0 = negro, 1 = blanco)')
    ax.set_title('Perfil de intensidad: RZI sigue mejor los saltos (bordes) del original')
    ax.legend()
    ax.grid(linestyle=':', alpha=0.6)
    guardar(fig, 'perfil_borde.png')

    # convergencia del modelo en esta imagen
    guardar(grafica_convergencia(res['historial'], res['omega'], res['error_datos']),
            'convergencia_rzi.png')

    # color
    rgb = data.astronaut().astype(np.float64) / 255.0
    H, W = rgb.shape[:2]
    lr_rgb = np.clip(resize_rgb_float(rgb, W // 4, H // 4, Image.BICUBIC), 0, 1)
    rzi_rgb = upscale(lr_rgb, 4)['imagen_recon']
    bic_rgb = resize_rgb_float(lr_rgb, W, H, Image.BICUBIC)
    nn_rgb = resize_rgb_float(lr_rgb, W, H, Image.NEAREST)
    y, x, lado = 40, 150, 200
    fig, ejes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (t, u) in zip(ejes, (('Original', rgb), ('Vecino mas proximo', nn_rgb),
                                 ('Bicubica', bic_rgb), ('RZI (luminancia + croma)', rzi_rgb))):
        ax.imshow(u[y:y + lado, x:x + lado], interpolation='nearest')
        ax.set_title(t)
        ax.axis('off')
    fig.suptitle('Ampliacion x4 a color (imagen pequena de 128x128 px)', fontsize=12)
    fig.tight_layout()
    guardar(fig, 'comparacion_color_x4.png')


# 4. Omega, benchmark

def omega_y_benchmark():
    print('Barrido de omega y benchmark...')
    tile = IMAGENES['camera'][128:384, 128:384]
    p = construir_problema(tile, 4)
    r = barrido_omega(p, tol=1e-3, max_iter=500, n_puntos=45)
    guardar(grafica_barrido_omega(r['omegas'], r['iteraciones'], r['omega_teorico'],
                                  r['omega_malla_completa']), 'barrido_omega.png')
    filas_omega = []
    for f in (2, 3, 4, 6, 8):
        pf = construir_problema(tile, f)
        rf = barrido_omega(pf, tol=1e-3, max_iter=800, n_puntos=45)
        filas_omega.append({'factor': f, 'omega_malla_completa': pf['omega_malla_completa'],
                            'omega_estimado': pf['omega_opt'],
                            'omega_empirico': rf['omega_empirico'],
                            'iter_con_estimado': min(rf['iteraciones'][int(np.argmin(
                                np.abs(np.array(rf['omegas']) - pf['omega_opt'])))], 800),
                            'iter_con_malla_completa': rf['iteraciones'][int(np.argmin(
                                np.abs(np.array(rf['omegas']) - pf['omega_malla_completa'])))]})
    with open(os.path.join(RES, 'omega_por_factor.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas_omega[0].keys()))
        w.writeheader()
        for fo in filas_omega:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in fo.items()})

    _calentamiento()
    bench = []
    for lado in (64, 128, 256):
        t = IMAGENES['camera'][:lado, :lado]
        bench.append(benchmark_numba_vs_python(construir_problema(t, 4), n_iter=20))
    guardar(grafica_benchmark(bench), 'benchmark.png')
    with open(os.path.join(RES, 'benchmark.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['malla', 'iteraciones', 'tiempo_python_s', 'tiempo_numba_s', 'aceleracion'])
        for b in bench:
            w.writerow([f"{b['forma'][0]}x{b['forma'][1]}", b['iteraciones'],
                        f"{b['tiempo_python']:.4f}", f"{b['tiempo_numba']:.6f}",
                        f"{b['aceleracion']:.0f}"])
    return r, filas_omega, bench


# 5. Figuras didacticas

def figuras_didacticas():
    print('Figuras didacticas...')
    # interpolaciones en 1D
    xs = np.arange(8)
    ys = np.array([0.2, 0.25, 0.2, 0.3, 0.85, 0.9, 0.85, 0.9])
    xf = np.linspace(0, 7, 400)
    nn = ys[np.clip(np.round(xf).astype(int), 0, 7)]
    lin = np.interp(xf, xs, ys)
    from scipy.interpolate import CubicSpline
    cub = CubicSpline(xs, ys)(xf)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xf, nn, color=ROJO, label='Vecino mas proximo (escalones)')
    ax.plot(xf, lin, color=AZUL, label='Lineal (rampas)')
    ax.plot(xf, cub, color=VERDE, label='Cubica (curva suave)')
    ax.plot(xs, ys, 'ko', markersize=8, label='Pixeles conocidos')
    ax.set_xlabel('Posicion')
    ax.set_ylabel('Brillo')
    ax.set_title('Tres formas de "rellenar" entre pixeles conocidos (en una fila)')
    ax.legend(fontsize=9)
    ax.grid(linestyle=':', alpha=0.6)
    guardar(fig, 'interpolaciones_1d.png')

    # estencil de 5 puntos y tablero rojo-negro
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4.6))
    ax = ejes[0]
    for i in range(5):
        for j in range(5):
            ax.add_patch(plt.Rectangle((j, 4 - i), 1, 1, fill=False, edgecolor='#BDBDBD'))
    celdas = {(2, 2): ('u[i,j]\npeso 4', '#FFCDD2'), (1, 2): ('u[i-1,j]\n-1', '#BBDEFB'),
              (3, 2): ('u[i+1,j]\n-1', '#BBDEFB'), (2, 1): ('u[i,j-1]\n-1', '#BBDEFB'),
              (2, 3): ('u[i,j+1]\n-1', '#BBDEFB')}
    for (i, j), (t, c) in celdas.items():
        ax.add_patch(plt.Rectangle((j, 4 - i), 1, 1, color=c))
        ax.text(j + 0.5, 4 - i + 0.5, t, ha='center', va='center', fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Estencil de 5 puntos (Laplaciano discreto)')
    ax = ejes[1]
    for i in range(8):
        for j in range(8):
            ax.add_patch(plt.Rectangle((j, 7 - i), 1, 1,
                                       color='#E53935' if (i + j) % 2 == 0 else '#424242'))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ordenamiento Rojo-Negro: cada rojo\nsolo tiene vecinos negros (y viceversa)')
    guardar(fig, 'estencil_rojo_negro.png')

    # mascara del problema clasico
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    n = 13
    m = np.zeros((n, n), dtype=bool)
    m[::4, ::4] = True
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    ax.imshow(m, cmap='Blues', vmin=-0.3, vmax=1.2)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='both', length=0, labelbottom=False, labelleft=False)
    ax.set_title('Pixeles conocidos (azul) con factor 4:\nla malla queda dividida en celdas')
    guardar(fig, 'mascara_factor4.png')

    # funcion de amplificacion s(t)
    t = np.linspace(0, 0.4, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    for beta, c in ((1.3, AZUL), (1.6, VERDE), (2.0, NARANJA)):
        ax.plot(t, 1 + (beta - 1) * t / (t + 0.05), color=c, label=f'beta = {beta}')
    ax.axvline(0.05, color=GRIS, linestyle=':', label='tau = 0.05')
    ax.set_xlabel('Tamano del gradiente |grad G| (que tan fuerte es el borde)')
    ax.set_ylabel('Factor de amplificacion s')
    ax.set_title('Bordes debiles casi no se tocan; bordes fuertes se refuerzan')
    ax.legend()
    ax.grid(linestyle=':', alpha=0.6)
    guardar(fig, 'funcion_amplificacion.png')

    # campo de gradientes amplificado sobre un borde
    img = IMAGENES['camera'][80:200, 220:340]
    gx, gy = amplified_guidance(img, beta=PARAMS_DEFECTO['beta'], tau=PARAMS_DEFECTO['tau'])
    fig, ejes = plt.subplots(1, 2, figsize=(9, 4.4))
    ejes[0].imshow(img, cmap='gray')
    ejes[0].set_title('Imagen')
    ejes[1].imshow(np.hypot(gx, gy), cmap='magma')
    ejes[1].set_title('Magnitud del campo v (bordes)')
    for ax in ejes:
        ax.axis('off')
    guardar(fig, 'campo_gradientes.png')

    # diagrama del ciclo del metodo
    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    textos = ['Imagen pequena f\n(h x w)', 'Inicio:\nbicubica u0',
              'Back-projection\nt = u + Up(f - Down(u))',
              'Poisson screened\n(4+lam)u - vecinos = -div(v) + lam t\n(SOR Red-Black + Numba)',
              'Resultado u\n(h*k x w*k)']
    xs = [0.07, 0.24, 0.45, 0.72, 0.93]
    cajas = [ax.text(xc, 0.62, t, ha='center', va='center', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor=AZUL))
             for xc, t in zip(xs, textos)]
    for c1, c2 in zip(cajas[:-1], cajas[1:]):
        ax.annotate('', xy=(0, 0.5), xycoords=c2, xytext=(1, 0.5), textcoords=c1,
                    arrowprops=dict(arrowstyle='->', lw=1.8))
    ax.annotate('', xy=(0.5, 0), xycoords=cajas[2], xytext=(0.5, 0), textcoords=cajas[3],
                arrowprops=dict(arrowstyle='->', lw=1.6, color=ROSA,
                                connectionstyle='arc3,rad=-0.4'))
    ax.text(0.585, 0.2, 'se repite "ciclos" veces (10 por defecto)', ha='center',
            color=ROSA, fontsize=10)
    guardar(fig, 'diagrama_metodo.png')


def main():
    t0 = time.time()
    filas = tabla_principal()
    resumen = resumen_ganancias(filas)
    rob = robustez()
    comparacion_visual()
    r, filas_omega, bench = omega_y_benchmark()
    figuras_didacticas()

    print('\nResumen (promedio de 4 imagenes):')
    for r_ in resumen:
        print(f"  x{r_['factor']} {r_['metodo']:<11} PSNR {r_['psnr']:.2f}  "
              f"SSIM {r_['ssim']:.4f}  GMSD {r_['gmsd']:.4f}  nitidez/HR {r_['nitidez_rel']:.3f}")
    print('Robustez:')
    for f in rob:
        print(f"  {f['degradacion']:<34} {f['ganancia_media_vs_bicubica']:+.2f} dB "
              f"(min {f['ganancia_min_vs_bicubica']:+.2f})")
    print('Omega:')
    for fo in filas_omega:
        print('  ', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in fo.items()})
    for b in bench:
        print(f"  benchmark {b['forma']}: x{b['aceleracion']:.0f}")
    print(f'Listo en {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
